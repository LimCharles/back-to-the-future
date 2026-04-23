"""Clone-Hidden HMM (CHMM) — ported from sukumar1612/Ctrl-G (distillation branch).

The model class is a direct port of ``distillation/chmm.py`` from the reference
repository. Two integration hooks have been added so CHMM fits the same
generate.py dispatch pattern as HMM/SOHMM:

* :meth:`CHMM.set_weights` — register per-token coefficients (matches
  ``HMM.set_weights`` / ``SOHMM.set_weights``).
* :meth:`CHMM.compute_backward_expectation` — returns the (T, H) backward
  expectation cache that :class:`CHMMGuidedLogitsProcessor` consumes.

These are the only additions; the forward/backward EM machinery is kept
intact so loaded checkpoints remain round-trippable with the reference
training pipeline.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

__all__ = ["CHMM", "collect_pair_codes_from_sequences"]


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(
    pf: torch.Tensor,
    pp: torch.Tensor,
    cp: torch.Tensor,
) -> torch.Tensor:
    ll = torch.amax(cp, dim=-1)
    ll = torch.where(torch.isfinite(ll), ll, torch.zeros_like(ll))

    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    pp = torch.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0)
    cp = torch.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    return matmul(ratio, cp)


def collect_pair_codes_from_sequences(
    input_ids: torch.Tensor | Sequence[Sequence[int]],
    vocab_size: int,
) -> torch.Tensor:
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if input_ids.shape[1] < 2:
        return torch.empty(0, dtype=torch.long)

    src = input_ids[:, :-1]
    dst = input_ids[:, 1:]
    observed = (src != -1) & (dst != -1)
    if not observed.any():
        return torch.empty(0, dtype=torch.long)

    pair_codes = src[observed] * vocab_size + dst[observed]
    return torch.unique(pair_codes, sorted=True).cpu()


@dataclass(frozen=True)
class BlockSpec:
    src_token: int
    dst_token: int
    src_start: int
    dst_start: int
    src_size: int
    dst_size: int
    value_start: int
    value_stop: int

    @property
    def src_stop(self) -> int:
        return self.src_start + self.src_size

    @property
    def dst_stop(self) -> int:
        return self.dst_start + self.dst_size

    @property
    def pair_code(self) -> int:
        raise RuntimeError("pair_code is stored externally, not in BlockSpec.")


class SparseTransitionTable(nn.Module):
    def __init__(
        self,
        clones_per_token: torch.Tensor,
        pair_codes: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        init_random: bool = True,
    ) -> None:
        super().__init__()

        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long)
        pair_codes = self._normalize_pair_codes(pair_codes, len(clones_per_token))
        specs = self._build_block_specs(clones_per_token, pair_codes)
        values = (
            self._random_transition_values(clones_per_token, specs, dtype)
            if init_random
            else torch.empty(specs[-1].value_stop if specs else 0, dtype=dtype)
        )

        self.register_buffer("pair_codes", pair_codes)
        self.register_buffer("transition_values", values)
        self.register_buffer(
            "block_value_start",
            torch.tensor([spec.value_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_src_start",
            torch.tensor([spec.src_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_dst_start",
            torch.tensor([spec.dst_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_src_size",
            torch.tensor([spec.src_size for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_dst_size",
            torch.tensor([spec.dst_size for spec in specs], dtype=torch.long),
        )

        self.block_specs = specs
        self.outgoing_block_ids = self._build_outgoing_lists(len(clones_per_token), specs)
        self.incoming_block_ids = self._build_incoming_lists(len(clones_per_token), specs)

    def block_index_for_pair_code(self, pair_code: int) -> int | None:
        if self.pair_codes.numel() == 0:
            return None

        query = torch.tensor([pair_code], device=self.pair_codes.device, dtype=self.pair_codes.dtype)
        position = int(torch.searchsorted(self.pair_codes, query).item())
        if position >= int(self.pair_codes.numel()):
            return None
        if int(self.pair_codes[position].item()) != pair_code:
            return None
        return position

    @staticmethod
    def _normalize_pair_codes(
        pair_codes: torch.Tensor | Sequence[int] | Sequence[tuple[int, int]] | None,
        vocab_size: int,
    ) -> torch.Tensor:
        if pair_codes is None:
            if vocab_size > 512:
                raise ValueError(
                    "pair_codes is required for large sparse CHMMs. "
                    "Use collect_pair_codes_from_sequences(...) first."
                )
            pair_codes = torch.arange(vocab_size * vocab_size, dtype=torch.long)
        elif torch.is_tensor(pair_codes):
            pair_codes = pair_codes.to(dtype=torch.long, device="cpu")
        else:
            first_item = next(iter(pair_codes), None)
            if first_item is None:
                pair_codes = torch.empty(0, dtype=torch.long)
            elif isinstance(first_item, tuple):
                pair_codes = torch.tensor(
                    [src * vocab_size + dst for src, dst in pair_codes],
                    dtype=torch.long,
                )
            else:
                pair_codes = torch.tensor(list(pair_codes), dtype=torch.long)

        if pair_codes.numel() == 0:
            return pair_codes

        pair_codes = torch.unique(pair_codes, sorted=True)
        if torch.any(pair_codes < 0) or torch.any(pair_codes >= vocab_size * vocab_size):
            raise ValueError("pair_codes contains an out-of-range token pair.")
        return pair_codes

    @staticmethod
    def _build_block_specs(
        clones_per_token: torch.Tensor,
        pair_codes: torch.Tensor,
    ) -> list[BlockSpec]:
        state_offsets = torch.zeros(len(clones_per_token) + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)

        specs: list[BlockSpec] = []
        value_cursor = 0
        vocab_size = len(clones_per_token)
        for code in pair_codes.tolist():
            src_token = code // vocab_size
            dst_token = code % vocab_size

            src_start = int(state_offsets[src_token].item())
            dst_start = int(state_offsets[dst_token].item())
            src_size = int(clones_per_token[src_token].item())
            dst_size = int(clones_per_token[dst_token].item())
            value_count = src_size * dst_size

            specs.append(
                BlockSpec(
                    src_token=src_token,
                    dst_token=dst_token,
                    src_start=src_start,
                    dst_start=dst_start,
                    src_size=src_size,
                    dst_size=dst_size,
                    value_start=value_cursor,
                    value_stop=value_cursor + value_count,
                )
            )
            value_cursor += value_count
        return specs

    @staticmethod
    def _build_outgoing_lists(vocab_size: int, specs: Sequence[BlockSpec]) -> list[list[int]]:
        outgoing = [[] for _ in range(vocab_size)]
        for idx, spec in enumerate(specs):
            outgoing[spec.src_token].append(idx)
        return outgoing

    @staticmethod
    def _build_incoming_lists(vocab_size: int, specs: Sequence[BlockSpec]) -> list[list[int]]:
        incoming = [[] for _ in range(vocab_size)]
        for idx, spec in enumerate(specs):
            incoming[spec.dst_token].append(idx)
        return incoming

    def _random_transition_values(
        self,
        clones_per_token: torch.Tensor,
        specs: Sequence[BlockSpec],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not specs:
            return torch.empty(0, dtype=dtype)

        values = torch.empty(specs[-1].value_stop, dtype=dtype)
        outgoing = self._build_outgoing_lists(len(clones_per_token), specs)

        for src_token, block_ids in enumerate(outgoing):
            if not block_ids:
                continue

            src_size = int(clones_per_token[src_token].item())
            total_dst = sum(specs[block_idx].dst_size for block_idx in block_ids)
            packed = torch.rand(src_size, total_dst, dtype=dtype)
            packed /= packed.sum(dim=1, keepdim=True)

            cursor = 0
            for block_idx in block_ids:
                spec = specs[block_idx]
                width = spec.dst_size
                self.block_view(values, spec)[:, :] = packed[:, cursor : cursor + width]
                cursor += width

        return values

    def block_view(self, flat_values: torch.Tensor, spec: BlockSpec) -> torch.Tensor:
        return flat_values[spec.value_start : spec.value_stop].view(spec.src_size, spec.dst_size)

    def block(self, block_idx: int) -> torch.Tensor:
        return self.block_view(self.transition_values, self.block_specs[block_idx])

    def empty_counts(self, device: torch.device | None = None) -> torch.Tensor:
        device = self.transition_values.device if device is None else device
        return torch.zeros(
            self.transition_values.shape[0],
            device=device,
            dtype=self.transition_values.dtype,
        )

    def update_from_dense(self, dense_transition: torch.Tensor) -> None:
        new_values = torch.zeros_like(self.transition_values)
        for spec in self.block_specs:
            dense_block = dense_transition[
                spec.src_start : spec.src_stop,
                spec.dst_start : spec.dst_stop,
            ]
            self.block_view(new_values, spec).copy_(dense_block)
        self.transition_values.copy_(new_values)

    def normalized_values_from_counts(
        self,
        transition_counts: torch.Tensor,
        pseudocount: float,
        hidden_states: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pseudocount < 0.0:
            raise ValueError("pseudocount must be non-negative.")

        transition_counts = transition_counts.to(
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )
        new_values = torch.zeros_like(self.transition_values)
        row_totals = torch.zeros(
            hidden_states,
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )

        group_codes = None
        size_base = 0
        if self.transition_values.numel() > 0:
            size_base = int(self.block_dst_size.max().item()) + 1
            group_codes = self.block_src_size * size_base + self.block_dst_size
            for code in torch.unique(group_codes):
                src_size = int((code // size_base).item())
                dst_size = int((code % size_base).item())
                block_ids = (group_codes == code).nonzero(as_tuple=False).squeeze(-1)
                starts = self.block_value_start[block_ids]
                value_offsets = starts[:, None] + torch.arange(
                    src_size * dst_size,
                    device=self.transition_values.device,
                )
                block_counts = transition_counts[value_offsets].view(-1, src_size, dst_size)
                row_counts = block_counts.sum(dim=2)
                row_offsets = self.block_src_start[block_ids, None] + torch.arange(
                    src_size,
                    device=self.transition_values.device,
                )
                row_totals.index_add_(0, row_offsets.reshape(-1), row_counts.reshape(-1))

        denom = row_totals + float(pseudocount) * hidden_states
        valid_rows = denom > 0.0
        safe_denom = torch.where(valid_rows, denom, torch.ones_like(denom))

        if group_codes is not None:
            for code in torch.unique(group_codes):
                src_size = int((code // size_base).item())
                dst_size = int((code % size_base).item())
                block_ids = (group_codes == code).nonzero(as_tuple=False).squeeze(-1)
                starts = self.block_value_start[block_ids]
                value_offsets = starts[:, None] + torch.arange(
                    src_size * dst_size,
                    device=self.transition_values.device,
                )
                block_counts = transition_counts[value_offsets].view(-1, src_size, dst_size)
                row_offsets = self.block_src_start[block_ids, None] + torch.arange(
                    src_size,
                    device=self.transition_values.device,
                )
                block_denom = safe_denom[row_offsets].unsqueeze(-1)
                new_values[value_offsets.reshape(-1)] = (block_counts / block_denom).reshape(-1)

        floor = torch.zeros_like(row_totals)
        if pseudocount > 0.0:
            floor = torch.where(
                valid_rows,
                torch.full_like(row_totals, float(pseudocount)) / safe_denom,
                floor,
            )

        return new_values, floor

    def to_dense(self, hidden_states: int) -> torch.Tensor:
        dense = torch.zeros(
            hidden_states,
            hidden_states,
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )
        for block_idx, spec in enumerate(self.block_specs):
            dense[
                spec.src_start : spec.src_stop,
                spec.dst_start : spec.dst_stop,
            ] = self.block(block_idx)
        return dense


class CHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        clones_per_token: list[int] | torch.Tensor,
        *,
        pair_codes: torch.Tensor | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        dtype: torch.dtype = torch.float32,
        init_random: bool = True,
    ) -> None:
        super().__init__()

        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long)
        if len(clones_per_token) != vocab_size:
            raise ValueError("One clone count is required for each token.")

        hidden_states = int(clones_per_token.sum().item())
        gamma = torch.log_softmax(torch.randn(hidden_states, dtype=dtype), dim=0)
        state_offsets = torch.zeros(vocab_size + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)

        self.register_buffer("clones_per_token", clones_per_token)
        self.register_buffer(
            "clone_to_token",
            torch.repeat_interleave(torch.arange(vocab_size, dtype=torch.long), clones_per_token),
        )
        self.register_buffer("state_offsets", state_offsets)
        self.register_buffer("gamma", gamma)
        self.register_buffer("transition_floor", torch.zeros(hidden_states, dtype=dtype))

        self.transitions = SparseTransitionTable(
            clones_per_token=clones_per_token,
            pair_codes=pair_codes,
            dtype=dtype,
            init_random=init_random,
        )

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.max_clones = int(clones_per_token.max().item())
        self.clone_size_values = sorted({int(size) for size in clones_per_token.tolist()})

        self._block_groups: list | None = None
        self._block_groups_device: torch.device | None = None
        self._max_block_chunk: int = 0

    @property
    def device(self) -> torch.device:
        return self.gamma.device

    @property
    def pair_codes(self) -> torch.Tensor:
        return self.transitions.pair_codes

    @property
    def alpha_exp(self) -> torch.Tensor:
        dense = self.transitions.to_dense(self.hidden_states)
        return dense + self.transition_floor.unsqueeze(1)

    def config_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "clones_per_token": self.clones_per_token.tolist(),
            "num_pair_blocks": int(self.pair_codes.numel()),
        }

    @torch.no_grad()
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config_dict(),
            "gamma": self.gamma.detach().cpu(),
            "pair_codes": self.pair_codes.detach().cpu(),
            "transition_values": self.transitions.transition_values.detach().cpu(),
            "transition_floor": self.transition_floor.detach().cpu(),
        }
        torch.save(payload, output_dir / "model.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as fout:
            json.dump(payload["config"], fout, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, map_location: str | torch.device | None = None) -> "CHMM":
        model_path = Path(model_path)
        payload = torch.load(model_path / "model.pt", map_location="cpu", weights_only=True)

        pair_codes = payload.get("pair_codes")
        if pair_codes is None and "alpha_exp" in payload:
            pair_codes = cls._pair_codes_from_dense(
                alpha_exp=payload["alpha_exp"],
                clones_per_token=torch.tensor(payload["config"]["clones_per_token"], dtype=torch.long),
                vocab_size=payload["config"]["vocab_size"],
            )

        model = cls(
            vocab_size=payload["config"]["vocab_size"],
            eos_token_id=payload["config"]["eos_token_id"],
            clones_per_token=payload["config"]["clones_per_token"],
            pair_codes=pair_codes,
            dtype=payload["gamma"].dtype,
            init_random=False,
        )

        if "transition_values" in payload:
            model.update_params(payload["transition_values"], payload["gamma"])
        else:
            model.update_params(payload["alpha_exp"], payload["gamma"])

        if "transition_floor" in payload:
            model.transition_floor.copy_(payload["transition_floor"].to(dtype=model.gamma.dtype))

        if map_location is not None:
            model = model.to(map_location)
        return model

    @staticmethod
    def _pair_codes_from_dense(
        alpha_exp: torch.Tensor,
        clones_per_token: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        state_offsets = torch.zeros(vocab_size + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)
        pair_codes = []
        for src_token in range(vocab_size):
            for dst_token in range(vocab_size):
                src_slice = slice(int(state_offsets[src_token]), int(state_offsets[src_token + 1]))
                dst_slice = slice(int(state_offsets[dst_token]), int(state_offsets[dst_token + 1]))
                if alpha_exp[src_slice, dst_slice].abs().sum().item() > 0:
                    pair_codes.append(src_token * vocab_size + dst_token)
        return torch.tensor(pair_codes, dtype=torch.long)

    @torch.no_grad()
    def update_params(
        self,
        transition_values: torch.Tensor,
        gamma: torch.Tensor,
        transition_floor: torch.Tensor | None = None,
    ) -> None:
        gamma = gamma.to(self.gamma.device, dtype=self.gamma.dtype)
        self.gamma.copy_(gamma)

        transition_values = transition_values.to(
            self.transitions.transition_values.device,
            dtype=self.transitions.transition_values.dtype,
        )
        if transition_values.ndim == 2:
            self.transitions.update_from_dense(transition_values)
        elif transition_values.ndim == 1:
            if transition_values.shape != self.transitions.transition_values.shape:
                raise ValueError("transition_values has the wrong sparse storage shape.")
            self.transitions.transition_values.copy_(transition_values)
        else:
            raise ValueError("transition_values must be either dense [H, H] or sparse flat [nnz].")

        if transition_floor is None:
            self.transition_floor.zero_()
        else:
            transition_floor = transition_floor.to(self.device, dtype=self.gamma.dtype)
            if transition_floor.shape != self.transition_floor.shape:
                raise ValueError("transition_floor has the wrong shape.")
            self.transition_floor.copy_(transition_floor)

        self._block_groups = None

    def empty_count_buffers(
        self,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device if device is None else device
        transition_counts = self.transitions.empty_counts(device=device)
        gamma_counts = torch.zeros(self.hidden_states, device=device, dtype=self.gamma.dtype)
        return transition_counts, gamma_counts

    @torch.no_grad()
    def update_from_counts(
        self,
        transition_counts: torch.Tensor,
        gamma_counts: torch.Tensor,
        pseudocount: float,
    ) -> None:
        new_transition_values, new_transition_floor = self.transitions.normalized_values_from_counts(
            transition_counts=transition_counts,
            pseudocount=pseudocount,
            hidden_states=self.hidden_states,
        )
        gamma_counts = gamma_counts.to(self.device, dtype=self.gamma.dtype)
        gamma_counts = gamma_counts + pseudocount / self.hidden_states
        gamma_counts = gamma_counts / gamma_counts.sum()
        self.update_params(new_transition_values, torch.log(gamma_counts), new_transition_floor)

    def _token_slice(self, token_id: int) -> slice:
        start = int(self.state_offsets[token_id].item())
        stop = int(self.state_offsets[token_id + 1].item())
        return slice(start, stop)

    def _effective_block(self, block_idx: int) -> torch.Tensor:
        spec = self.transitions.block_specs[block_idx]
        floor = self.transition_floor[spec.src_start : spec.src_stop].unsqueeze(1)
        return self.transitions.block(block_idx) + floor

    def _floor_block(self, source_slice: slice, dest_slice: slice) -> torch.Tensor:
        src_size = source_slice.stop - source_slice.start
        dst_size = dest_slice.stop - dest_slice.start
        floor = self.transition_floor[source_slice].unsqueeze(1)
        return floor.expand(src_size, dst_size)

    def _floor_project(self, source_slice: slice, child_messages: torch.Tensor) -> torch.Tensor:
        floor = torch.log(self.transition_floor[source_slice]).unsqueeze(1)
        child_logsum = torch.logsumexp(child_messages, dim=0, keepdim=True)
        return floor + child_logsum

    def _stable_project(self, transition: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        msg_max = torch.amax(messages, dim=0, keepdim=True)
        msg_max = torch.where(torch.isfinite(msg_max), msg_max, torch.zeros_like(msg_max))
        probs = torch.exp(messages - msg_max)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        out = matmul(transition, probs)
        out = torch.log(out) + msg_max
        return out

    def _observed_pair_block_indices(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_codes = curr_tokens * self.vocab_size + next_tokens
        num_blocks = int(self.transitions.pair_codes.numel())
        if num_blocks == 0:
            return torch.zeros_like(pair_codes), torch.zeros_like(pair_codes, dtype=torch.bool)

        positions = torch.searchsorted(self.transitions.pair_codes, pair_codes)
        in_range = positions < num_blocks
        block_idx = torch.clamp(positions, max=num_blocks - 1)
        matched = in_range & (self.transitions.pair_codes[block_idx] == pair_codes)
        return block_idx, matched

    def _compact_size_groups(
        self,
        src_sizes: torch.Tensor,
        dst_sizes: torch.Tensor,
    ) -> Iterable[tuple[int, int, torch.Tensor]]:
        for src_size in self.clone_size_values:
            src_mask = src_sizes == src_size
            for dst_size in self.clone_size_values:
                cols = (src_mask & (dst_sizes == dst_size)).nonzero(as_tuple=False).squeeze(-1)
                if cols.numel() > 0:
                    yield src_size, dst_size, cols

    def _compact_token_size_groups(
        self,
        token_sizes: torch.Tensor,
    ) -> Iterable[tuple[int, torch.Tensor]]:
        for size in self.clone_size_values:
            cols = (token_sizes == size).nonzero(as_tuple=False).squeeze(-1)
            if cols.numel() > 0:
                yield size, cols

    def _compact_effective_blocks(
        self,
        curr_tokens: torch.Tensor,
        cols: torch.Tensor,
        block_idx: torch.Tensor,
        matched: torch.Tensor,
        src_size: int,
        dst_size: int,
    ) -> torch.Tensor:
        device = curr_tokens.device
        dtype = self.transitions.transition_values.dtype
        blocks = torch.zeros(
            (cols.shape[0], src_size, dst_size),
            device=device,
            dtype=dtype,
        )

        local_matched = matched[cols]
        if local_matched.any():
            local_rows = local_matched.nonzero(as_tuple=False).squeeze(-1)
            local_block_idx = block_idx[cols[local_rows]]
            starts = self.transitions.block_value_start[local_block_idx]
            offsets = starts[:, None] + torch.arange(src_size * dst_size, device=device)
            values = self.transitions.transition_values[offsets].view(-1, src_size, dst_size)
            blocks[local_rows] = values

        src_starts = self.state_offsets[curr_tokens[cols]]
        src_offsets = src_starts[:, None] + torch.arange(src_size, device=device)
        floor = self.transition_floor[src_offsets]
        return blocks + floor.unsqueeze(-1)

    def _compact_project_observed_step(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        child_messages: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = curr_tokens.shape[0]
        parent_messages = torch.full(
            (batch_size, self.max_clones),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        src_sizes = self.clones_per_token[curr_tokens]
        dst_sizes = self.clones_per_token[next_tokens]
        block_idx, matched = self._observed_pair_block_indices(curr_tokens, next_tokens)

        for src_size, dst_size, cols in self._compact_size_groups(src_sizes, dst_sizes):
            blocks = self._compact_effective_blocks(
                curr_tokens=curr_tokens,
                cols=cols,
                block_idx=block_idx,
                matched=matched,
                src_size=src_size,
                dst_size=dst_size,
            )
            child = child_messages[cols, :dst_size]
            child_max = torch.amax(child, dim=1, keepdim=True)
            child_max = torch.where(torch.isfinite(child_max), child_max, torch.zeros_like(child_max))
            child_probs = torch.exp(child - child_max)
            child_probs = torch.nan_to_num(child_probs, nan=0.0, posinf=0.0, neginf=0.0)
            projected = torch.bmm(blocks, child_probs.unsqueeze(-1)).squeeze(-1)
            parent_messages[cols, :src_size] = torch.log(projected) + child_max

        return parent_messages

    def forward_observed_compact(self, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if torch.any(input_ids == -1):
            raise ValueError("forward_observed_compact requires fully observed input_ids.")

        batch_size, seq_len = input_ids.shape
        messages: list[torch.Tensor] = [
            torch.empty(0, device=self.device, dtype=self.gamma.dtype)
            for _ in range(seq_len)
        ]

        last_tokens = input_ids[:, -1]
        last_sizes = self.clones_per_token[last_tokens]
        last_messages = torch.full(
            (batch_size, self.max_clones),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        for size, cols in self._compact_token_size_groups(last_sizes):
            last_messages[cols, :size] = 0.0
        messages[-1] = last_messages

        for t in range(seq_len - 2, -1, -1):
            messages[t] = self._compact_project_observed_step(
                curr_tokens=input_ids[:, t],
                next_tokens=input_ids[:, t + 1],
                child_messages=messages[t + 1],
            )

        first_tokens = input_ids[:, 0]
        first_sizes = self.clones_per_token[first_tokens]
        first_starts = self.state_offsets[first_tokens]
        loglik = torch.full(
            (batch_size,),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        for size, cols in self._compact_token_size_groups(first_sizes):
            offsets = first_starts[cols, None] + torch.arange(size, device=self.device)
            loglik[cols] = torch.logsumexp(self.gamma[offsets] + messages[0][cols, :size], dim=1)

        return messages, loglik

    def loglikelihood(self, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        ll = torch.tensor([0.0], device=self.device, dtype=self.gamma.dtype)
        for batch_idx in range(0, input_ids.shape[0], batch_size):
            input_ids_batch_cpu = input_ids[batch_idx : batch_idx + batch_size]
            has_missing = torch.any(input_ids_batch_cpu == -1)
            input_ids_batch = input_ids_batch_cpu.to(self.device)
            if not has_missing:
                _, loglik = self.forward_observed_compact(input_ids_batch)
                ll += torch.sum(loglik)
            else:
                raise NotImplementedError(
                    "CHMM.loglikelihood with missing tokens (-1) is not wired "
                    "in this inference-only fork; feed fully observed sequences."
                )
        return ll

    # ------------------------------------------------------------------
    # TRACE integration hooks (added for back-to-the-future fork)
    # ------------------------------------------------------------------

    def _ensure_block_groups(self, max_block_values_per_chunk: int = 4_000_000) -> None:
        """Lazily build (src_size, dst_size)-grouped block index tables.

        Used by ``_transition_matvec`` and ``compute_backward_expectation``
        for fast sparse matrix-vector products over the block-structured
        transition table. Cached per-device.
        """
        if (
            self._block_groups is not None
            and self._block_groups_device == self.device
            and self._max_block_chunk == max_block_values_per_chunk
        ):
            return

        grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for spec in self.transitions.block_specs:
            grouped[(spec.src_size, spec.dst_size)].append(
                (spec.value_start, spec.src_start, spec.dst_start)
            )

        device = self.device
        block_groups = []
        for (src_size, dst_size), starts in grouped.items():
            value_starts, src_starts, dst_starts = zip(*starts)
            block_groups.append(
                (
                    src_size,
                    dst_size,
                    torch.tensor(value_starts, device=device, dtype=torch.long),
                    torch.tensor(src_starts, device=device, dtype=torch.long),
                    torch.tensor(dst_starts, device=device, dtype=torch.long),
                    torch.arange(src_size * dst_size, device=device),
                    torch.arange(src_size, device=device),
                    torch.arange(dst_size, device=device),
                )
            )
        self._block_groups = block_groups
        self._block_groups_device = device
        self._max_block_chunk = max_block_values_per_chunk

    def _transition_matvec(
        self,
        dest_values: torch.Tensor,
        max_block_values_per_chunk: int = 4_000_000,
    ) -> torch.Tensor:
        """Sparse block-structured matrix-vector product ``A @ dest_values``.

        ``out[s] = transition_floor[s] * dest_values.sum()
                 + sum_{block: s in block.src} block_values @ dest_values[block.dst]``

        Lifted from the reference CHMMGuidedLogitsProcessor so the backward
        expectation can live on the model.
        """
        self._ensure_block_groups(max_block_values_per_chunk)
        transition_values = self.transitions.transition_values
        out = self.transition_floor * dest_values.sum()
        for (
            src_size,
            dst_size,
            value_starts,
            src_starts,
            dst_starts,
            value_local_offsets,
            src_local_offsets,
            dst_local_offsets,
        ) in self._block_groups:
            values_per_block = src_size * dst_size
            chunk_size = max(1, max_block_values_per_chunk // values_per_block)
            for start in range(0, int(value_starts.numel()), chunk_size):
                stop = min(start + chunk_size, int(value_starts.numel()))
                local_value_starts = value_starts[start:stop]
                local_src_starts = src_starts[start:stop]
                local_dst_starts = dst_starts[start:stop]
                value_offsets = local_value_starts[:, None] + value_local_offsets
                block_values = transition_values[value_offsets].view(
                    -1, src_size, dst_size
                )
                dst_offsets = local_dst_starts[:, None] + dst_local_offsets
                projected = torch.bmm(
                    block_values,
                    dest_values[dst_offsets].unsqueeze(-1),
                ).squeeze(-1)
                src_offsets = local_src_starts[:, None] + src_local_offsets
                out.index_add_(0, src_offsets.reshape(-1), projected.reshape(-1))
        return out

    def set_weights(self, weights_tensor: torch.Tensor) -> None:
        """Register per-token coefficients.

        Mirrors ``HMM.set_weights`` / ``SOHMM.set_weights`` so the same
        generate.py dispatch pattern (load → set_weights →
        compute_backward_expectation → processor) works for CHMM.
        """
        if weights_tensor.shape != (self.vocab_size,):
            raise ValueError(
                f"weights_tensor must have shape ({self.vocab_size},), "
                f"got {tuple(weights_tensor.shape)}"
            )
        weights_tensor = weights_tensor.to(self.device, dtype=self.gamma.dtype)
        exp_weights = torch.exp(weights_tensor)
        clone_exp_weights = exp_weights[self.clone_to_token]

        existing = dict(self.named_buffers())
        for name, value in (
            ("weights_tensor", weights_tensor),
            ("exp_weights", exp_weights),
            ("clone_exp_weights", clone_exp_weights),
        ):
            if name in existing:
                getattr(self, name).resize_as_(value).copy_(value)
            else:
                self.register_buffer(name, value)

    def compute_backward_expectation(self, T: int) -> torch.Tensor:
        """Backward expectation cache E[exp(Σ w(x_i)) | z_t] for t in [0, T).

        Returns ``(T, H)`` tensor consumed by
        :class:`CHMMGuidedLogitsProcessor` as its ``expectation_cache``.
        Requires :meth:`set_weights` to have been called first.
        """
        if not hasattr(self, "clone_exp_weights"):
            raise RuntimeError(
                "CHMM.compute_backward_expectation requires set_weights() first"
            )
        if T <= 0:
            raise ValueError("T must be positive")
        device = self.device
        dtype = self.gamma.dtype
        cache = torch.empty((T, self.hidden_states), device=device, dtype=dtype)
        future = torch.ones(self.hidden_states, device=device, dtype=dtype)
        cache[T - 1] = future
        for t in range(T - 2, -1, -1):
            weighted_dest = self.clone_exp_weights * future
            future = self._transition_matvec(weighted_dest)
            future = torch.nan_to_num(future, nan=0.0, posinf=1e12, neginf=0.0)
            cache[t] = future
        return cache
