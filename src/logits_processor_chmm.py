"""TRACE decode kernel for CHMM.

Adapted from ``distillation/chmm_logits_processor.py`` in
sukumar1612/Ctrl-G so the constructor/configure_for_prompts contract
lines up with :class:`HmmGuidedLogitsProcessor` and
:class:`SOHmmGuidedLogitsProcessor`:

* The backward-expectation cache is built externally via
  ``CHMM.compute_backward_expectation(T=max_len)`` and passed in.
* Per-token coefficients live on the CHMM model
  (``set_weights(weights_tensor)``) instead of being a processor-ctor arg.
* ``configure_for_prompts`` takes only a pre-expanded ``prompt_batch``.
* First-step EAP dump emits the same ``.npz`` keys as hmm1/hmm2 so the
  transformation-distribution plot reads all three uniformly.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer

from src.chmm import CHMM

__all__ = ["CHMMGuidedLogitsProcessor"]


def _compute_eap_for_dump_chmm(
    pred_state: torch.Tensor,                # (B, H)
    future: torch.Tensor,                    # (H,)
    clone_to_token: torch.Tensor,            # (H,)
    vocab_size: int,
    exp_weights: torch.Tensor,               # (V,)
    product_generated_weight: torch.Tensor,  # (B,)
    a: float,
    epsilon: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """First-step EAP dump helper — same (pre, post) contract as hmm1/hmm2.

    Returns (eap_pre_transform, eap_post_transform), both shape (B, V).
    Only invoked once on ``generation_step == 0`` when ``dump_eap_path``
    is set, so the extra compute is trivial.
    """
    with torch.no_grad():
        batch = pred_state.shape[0]
        token_index = clone_to_token.unsqueeze(0).expand(batch, -1)
        token_prob = torch.zeros(
            batch, vocab_size, device=pred_state.device, dtype=pred_state.dtype
        )
        token_future = torch.zeros_like(token_prob)
        token_prob.scatter_add_(1, token_index, pred_state)
        token_future.scatter_add_(1, token_index, pred_state * future.unsqueeze(0))

        conditional_future = token_future / token_prob.clamp_min(epsilon)
        expectation = (
            conditional_future
            * exp_weights.unsqueeze(0)
            * product_generated_weight.unsqueeze(1)
        )
        eap_pre = torch.clamp(expectation, epsilon, 1.0 - epsilon)
        logit_eap = torch.log(eap_pre / (1.0 - eap_pre + epsilon) + epsilon)
        eap_post = torch.sigmoid(a * logit_eap)
        return eap_pre, eap_post


class CHMMGuidedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        hmm_model: Any,
        expectation_cache: torch.Tensor,
        a: float = 1.0,
        tokenizer: PreTrainedTokenizer | None = None,
        epsilon: float = 1e-12,
        dump_eap_path: Optional[str] = None,
        max_block_values_per_chunk: int = 4_000_000,
    ) -> None:
        if not isinstance(hmm_model, CHMM):
            required = (
                "transitions",
                "clone_to_token",
                "state_offsets",
                "gamma",
                "transition_floor",
                "exp_weights",
                "clone_exp_weights",
            )
            for name in required:
                if not hasattr(hmm_model, name):
                    raise AttributeError(f"CHMM model missing required attribute: {name}")
        if not hasattr(hmm_model, "exp_weights"):
            raise RuntimeError(
                "CHMMGuidedLogitsProcessor: call chmm.set_weights(weights_tensor) "
                "before constructing the processor."
            )

        self.hmm_model = hmm_model.eval()
        self._model_device = hmm_model.device
        self.dtype = torch.float32
        self.a = float(a)
        self.epsilon = float(epsilon)
        self.tokenizer = tokenizer
        self.dump_eap_path = dump_eap_path
        self._eap_dumped = False
        self.max_block_values_per_chunk = int(max_block_values_per_chunk)

        self.vocab_size = int(hmm_model.vocab_size)
        self.hidden_states = int(hmm_model.hidden_states)
        self.block_specs = list(hmm_model.transitions.block_specs)
        self.pair_to_block = {
            (spec.src_token, spec.dst_token): idx
            for idx, spec in enumerate(self.block_specs)
        }
        self.state_offsets = hmm_model.state_offsets.detach().cpu().tolist()

        self.transition_values = hmm_model.transitions.transition_values.detach().to(
            device=self._model_device, dtype=self.dtype
        )
        self.transition_floor = hmm_model.transition_floor.detach().to(
            device=self._model_device, dtype=self.dtype
        )
        self.gamma_probs = torch.softmax(
            hmm_model.gamma.detach().to(device=self._model_device, dtype=self.dtype),
            dim=0,
        )
        self.clone_to_token = hmm_model.clone_to_token.detach().to(
            device=self._model_device, dtype=torch.long
        )
        self.outgoing_groups_by_token = self._build_outgoing_groups_by_token()

        self.exp_weights = hmm_model.exp_weights.detach().to(
            device=self._model_device, dtype=self.dtype
        )
        self.clone_exp_weights = hmm_model.clone_exp_weights.detach().to(
            device=self._model_device, dtype=self.dtype
        )

        if expectation_cache.shape[1] != self.hidden_states:
            raise ValueError(
                f"expectation_cache second dim {expectation_cache.shape[1]} "
                f"does not match hidden_states {self.hidden_states}"
            )
        self.expectation_cache = expectation_cache.to(
            device=self._model_device, dtype=self.dtype
        )

        self.alpha_prev: torch.Tensor | None = None
        self.current_tokens: torch.Tensor | None = None
        self.product_generated_weight: torch.Tensor | None = None
        self.prompt_len = -1
        self.generation_step = 0
        self.is_configured_for_prompt = False
        self.zero_probability_resets = 0

    def _build_outgoing_groups_by_token(self):
        grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
        for spec in self.block_specs:
            grouped[(spec.src_token, spec.src_size, spec.dst_size)].append(
                (spec.value_start, spec.dst_start)
            )
        groups_by_token: dict[int, list] = defaultdict(list)
        device = self._model_device
        for (src_token, src_size, dst_size), starts in grouped.items():
            value_starts, dst_starts = zip(*starts)
            groups_by_token[src_token].append(
                (
                    src_size,
                    dst_size,
                    torch.tensor(value_starts, device=device, dtype=torch.long),
                    torch.tensor(dst_starts, device=device, dtype=torch.long),
                    torch.arange(src_size * dst_size, device=device),
                    torch.arange(dst_size, device=device),
                )
            )
        return dict(groups_by_token)

    def _token_slice(self, token_id: int) -> slice:
        return slice(self.state_offsets[token_id], self.state_offsets[token_id + 1])

    def _block(self, block_idx: int) -> torch.Tensor:
        spec = self.block_specs[block_idx]
        return self.transition_values[spec.value_start : spec.value_stop].view(
            spec.src_size, spec.dst_size
        )

    def _current_floor_weight(
        self,
        alpha: torch.Tensor,
        current_tokens: torch.Tensor,
    ) -> torch.Tensor:
        floor_weight = torch.zeros(
            alpha.shape[0], device=self._model_device, dtype=self.dtype
        )
        for src_token in torch.unique(current_tokens).detach().cpu().tolist():
            rows = (current_tokens == src_token).nonzero(as_tuple=False).squeeze(-1)
            source_slice = self._token_slice(int(src_token))
            floor_weight[rows] = (
                alpha[rows, source_slice] * self.transition_floor[source_slice]
            ).sum(dim=1)
        return floor_weight

    def _observe_tokens(
        self,
        alpha: torch.Tensor,
        current_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        next_tokens = next_tokens.to(device=self._model_device, dtype=torch.long)
        next_alpha = torch.zeros_like(alpha)
        floor_weight = self._current_floor_weight(alpha, current_tokens)

        for src_token in torch.unique(current_tokens).detach().cpu().tolist():
            src_rows = (current_tokens == src_token).nonzero(as_tuple=False).squeeze(-1)
            source_slice = self._token_slice(int(src_token))
            for dst_token in torch.unique(next_tokens[src_rows]).detach().cpu().tolist():
                rows = src_rows[next_tokens[src_rows] == int(dst_token)]
                dest_slice = self._token_slice(int(dst_token))
                next_alpha[rows, dest_slice] += floor_weight[rows, None]

                block_idx = self.pair_to_block.get((int(src_token), int(dst_token)))
                if block_idx is not None:
                    next_alpha[rows, dest_slice] += alpha[rows, source_slice].matmul(
                        self._block(block_idx)
                    )

        row_sums = next_alpha.sum(dim=1, keepdim=True)
        bad_rows = (row_sums.squeeze(1) <= self.epsilon).nonzero(as_tuple=False).squeeze(-1)
        if bad_rows.numel() > 0:
            self.zero_probability_resets += int(bad_rows.numel())
            for row_idx in bad_rows.detach().cpu().tolist():
                token_id = int(next_tokens[row_idx].item())
                dest_slice = self._token_slice(token_id)
                width = dest_slice.stop - dest_slice.start
                next_alpha[row_idx, dest_slice] = 1.0 / max(width, 1)
            row_sums = next_alpha.sum(dim=1, keepdim=True)

        return next_alpha / row_sums.clamp_min(self.epsilon)

    def _forward_observed_tokens(
        self, token_ids: List[int]
    ) -> Tuple[torch.Tensor, int]:
        if not token_ids:
            token_ids = [int(self.hmm_model.eos_token_id)]
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"token id {token_id} is outside the CHMM vocabulary"
                )

        first_token = int(token_ids[0])
        first_slice = self._token_slice(first_token)
        alpha = torch.zeros(
            self.hidden_states, device=self._model_device, dtype=self.dtype
        )
        alpha[first_slice] = self.gamma_probs[first_slice]
        if alpha.sum() <= self.epsilon:
            width = first_slice.stop - first_slice.start
            alpha[first_slice] = 1.0 / max(width, 1)
        else:
            alpha = alpha / alpha.sum()

        current_token = first_token
        for token_id in token_ids[1:]:
            current_tensor = torch.tensor(
                [current_token], device=self._model_device, dtype=torch.long
            )
            next_tensor = torch.tensor(
                [int(token_id)], device=self._model_device, dtype=torch.long
            )
            alpha = self._observe_tokens(
                alpha.unsqueeze(0), current_tensor, next_tensor
            ).squeeze(0)
            current_token = int(token_id)
        return alpha, current_token

    def configure_for_prompts(self, prompt_batch: torch.Tensor) -> None:
        """Initialise forward-α for each row in a pre-expanded prompt batch.

        generate.py pre-expands prompts via
        ``prompt_ids.repeat_interleave(num_to_generate, dim=0)`` before
        calling here, matching the hmm1/hmm2 contract. Left-pad tokens in
        the row are walked through forward — same assumption the other
        two processors already make.
        """
        prompt_batch = prompt_batch.to(device=self._model_device, dtype=torch.long)

        alphas, current_tokens = [], []
        for row_idx in range(prompt_batch.shape[0]):
            token_ids = [int(tid) for tid in prompt_batch[row_idx].detach().cpu().tolist()]
            alpha, current_token = self._forward_observed_tokens(token_ids)
            alphas.append(alpha)
            current_tokens.append(current_token)

        self.alpha_prev = torch.stack(alphas, dim=0)
        self.current_tokens = torch.tensor(
            current_tokens, device=self._model_device, dtype=torch.long
        )
        self.product_generated_weight = torch.ones(
            prompt_batch.shape[0], device=self._model_device, dtype=self.dtype
        )
        self.prompt_len = int(prompt_batch.shape[1])
        self.generation_step = 0
        self.is_configured_for_prompt = True

    def _predict_next_state(self) -> torch.Tensor:
        if self.alpha_prev is None or self.current_tokens is None:
            raise RuntimeError("Processor is not configured")

        alpha = self.alpha_prev
        current_tokens = self.current_tokens
        pred = torch.zeros_like(alpha)
        floor_weight = self._current_floor_weight(alpha, current_tokens)
        if torch.any(floor_weight > 0):
            pred += floor_weight[:, None]

        for src_token in torch.unique(current_tokens).detach().cpu().tolist():
            rows = (current_tokens == src_token).nonzero(as_tuple=False).squeeze(-1)
            source_slice = self._token_slice(int(src_token))
            source_alpha = alpha[rows, source_slice]
            for (
                src_size,
                dst_size,
                value_starts,
                dst_starts,
                value_local_offsets,
                dst_local_offsets,
            ) in self.outgoing_groups_by_token.get(int(src_token), []):
                values_per_block = src_size * dst_size
                chunk_size = max(1, self.max_block_values_per_chunk // values_per_block)
                for start in range(0, int(value_starts.numel()), chunk_size):
                    stop = min(start + chunk_size, int(value_starts.numel()))
                    local_value_starts = value_starts[start:stop]
                    local_dst_starts = dst_starts[start:stop]
                    value_offsets = local_value_starts[:, None] + value_local_offsets
                    block_values = self.transition_values[value_offsets].view(
                        -1, src_size, dst_size
                    )
                    projected = torch.einsum(
                        "bs,nsd->bnd", source_alpha, block_values
                    )
                    dst_offsets = local_dst_starts[:, None] + dst_local_offsets
                    row_offsets = rows[:, None, None].expand(
                        -1, projected.shape[1], dst_size
                    )
                    col_offsets = dst_offsets[None, :, :].expand(
                        rows.shape[0], -1, -1
                    )
                    pred.index_put_(
                        (row_offsets.reshape(-1), col_offsets.reshape(-1)),
                        projected.reshape(-1),
                        accumulate=True,
                    )

        row_sums = pred.sum(dim=1, keepdim=True)
        bad_rows = (row_sums.squeeze(1) <= self.epsilon).nonzero(as_tuple=False).squeeze(-1)
        if bad_rows.numel() > 0:
            self.zero_probability_resets += int(bad_rows.numel())
            pred[bad_rows] = self.gamma_probs.unsqueeze(0)
            row_sums = pred.sum(dim=1, keepdim=True)
        return pred / row_sums.clamp_min(self.epsilon)

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            if not self.is_configured_for_prompt:
                raise RuntimeError(
                    "CHMMGuidedLogitsProcessor.configure_for_prompts() must "
                    "be called before generation."
                )
            if scores.shape[-1] != self.vocab_size:
                raise ValueError(
                    f"LM vocab size {scores.shape[-1]} does not match CHMM "
                    f"vocab size {self.vocab_size}"
                )
            assert self.alpha_prev is not None
            assert self.current_tokens is not None
            assert self.product_generated_weight is not None

            input_ids = input_ids.to(device=self._model_device, dtype=torch.long)
            while input_ids.shape[1] > self.prompt_len + self.generation_step:
                new_pos = self.prompt_len + self.generation_step
                new_tokens = input_ids[:, new_pos]
                self.alpha_prev = self._observe_tokens(
                    self.alpha_prev, self.current_tokens, new_tokens
                )
                self.current_tokens = new_tokens
                self.product_generated_weight *= self.exp_weights[new_tokens]
                self.generation_step += 1

            if self.generation_step >= self.expectation_cache.shape[0]:
                return scores

            pred_state = self._predict_next_state()
            future = self.expectation_cache[self.generation_step]

            if self.dump_eap_path and self.generation_step == 0 and not self._eap_dumped:
                eap_pre, eap_post = _compute_eap_for_dump_chmm(
                    pred_state=pred_state,
                    future=future,
                    clone_to_token=self.clone_to_token,
                    vocab_size=self.vocab_size,
                    exp_weights=self.exp_weights,
                    product_generated_weight=self.product_generated_weight,
                    a=self.a,
                    epsilon=self.epsilon,
                )
                np.savez(
                    self.dump_eap_path,
                    eap_pre_transform=eap_pre.cpu().numpy(),
                    eap_post_transform=eap_post.cpu().numpy(),
                )
                self._eap_dumped = True

            token_index = self.clone_to_token.unsqueeze(0).expand(
                pred_state.shape[0], -1
            )
            token_prob = torch.zeros(
                pred_state.shape[0],
                self.vocab_size,
                device=self._model_device,
                dtype=self.dtype,
            )
            token_future = torch.zeros_like(token_prob)
            token_prob.scatter_add_(1, token_index, pred_state)
            token_future.scatter_add_(
                1, token_index, pred_state * future.unsqueeze(0)
            )

            conditional_future = token_future / token_prob.clamp_min(self.epsilon)
            expectation = (
                conditional_future
                * self.exp_weights.unsqueeze(0)
                * self.product_generated_weight.unsqueeze(1)
            )
            expectation = torch.clamp(expectation, self.epsilon, 1.0 - self.epsilon)
            logit_expectation = torch.log(
                expectation / (1.0 - expectation + self.epsilon) + self.epsilon
            )
            guidance_weight = torch.sigmoid(self.a * logit_expectation)

            lm_probs = torch.softmax(
                scores.to(device=self._model_device, dtype=self.dtype), dim=-1
            )
            adjusted = lm_probs * guidance_weight
            adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(
                self.epsilon
            )
            return torch.log(adjusted.clamp_min(self.epsilon)).to(
                device=scores.device, dtype=scores.dtype
            )
