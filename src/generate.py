#!/usr/bin/env python
import os
import sys
import math
import json
import argparse
from typing import List, Dict, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

# Determine project root and add to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)

# Local imports
from src import utils
from src.hmm import HMM
from src.sohmm import SOHMM
from src.chmm import CHMM
from src.logits_processor import HmmGuidedLogitsProcessor
from src.logits_processor_sohmm import SOHmmGuidedLogitsProcessor
from src.logits_processor_chmm import CHMMGuidedLogitsProcessor

def set_seed(seed: int, n_gpu: int):
    """Set random seed for reproducibility across PyTorch and CUDA."""
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="HMM-guided text generation.")
    parser.add_argument("--model_path", type=str, default="gpt2-large", help="HF model name/path for generation")
    parser.add_argument("--hmm_model_path", type=str, default="models/hmm_gpt2-large_uncon_seq-len-32_4096_10M", help="Path to the trained HMM directory")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.jsonl", help="Prompt file (JSONL)")
    parser.add_argument("--weights_path", type=str, default="data/coefficients.csv", help="Path to weights CSV file")
    parser.add_argument("--a", type=float, default=1.0, help="Strength of HMM guidance, often 0~2.0. 0 is no guidance, 1 is default.")
    parser.add_argument("--baseline", action="store_true", help="Generate baseline (no HMM guidance) alongside TRACE")
    parser.add_argument("--max_len", type=int, default=20, help="Max new tokens to generate")
    parser.add_argument("--num_generations", type=int, default=25, help="Generations per prompt")
    parser.add_argument("--generation_batch_size", type=int, default=5, help="Sequences per HF generate call")
    parser.add_argument("--prompt_batch_size", type=int, default=1, help="Prompts processed together")
    parser.add_argument("--hmm_variant", type=str, default="hmm1",
                        choices=["hmm1", "hmm2", "chmm"],
                        help="HMM variant: hmm1=first-order HMM, hmm2=second-order HMM (SOHMM/SHMM), chmm=clone-hidden HMM")
    parser.add_argument("--no_decode_transform", action="store_true",
                        help="Skip sigmoid-logit reshaping of EAP at decode time (hmm1 only)")
    parser.add_argument("--dump_eap_path", type=str, default=None,
                        help="Path to dump first-step per-token EAP as .npz (hmm1/hmm2/chmm)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Setup paths
    args.prompts_path = os.path.join(PROJECT_ROOT, args.prompts_path)
    args.weights_path = os.path.join(PROJECT_ROOT, args.weights_path)
    
    # Derive output CSV path relative to project root
    if args.baseline:
        output_path = os.path.join(
            PROJECT_ROOT,
            f"results/generated/comparison_{args.hmm_variant}_a{args.a}_generated.csv"
        )
    else:
        output_path = os.path.join(
            PROJECT_ROOT,
            f"results/generated/detox_{args.hmm_variant}_a{args.a}_generated.csv"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Setup device and seed
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.seed, torch.cuda.device_count())

    # Load generation model and tokenizer
    print(f"Loading generation model '{args.model_path}' …")
    gen_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    gen_tokenizer.pad_token = gen_tokenizer.pad_token or gen_tokenizer.eos_token

    #
    # LOAD HMM MODEL AND CONFIGURE WEIGHTS (variant-aware dispatch)
    #
    hmm_processor = None
    if not args.baseline or args.a > 0:  # Load HMM unless pure baseline mode
        # Weights file is required for any variant with guidance
        if not os.path.exists(args.weights_path):
            sys.exit(f"Missing weights file: {args.weights_path}")

        if args.hmm_variant == "hmm1":
            print(f"Loading first-order HMM from '{args.hmm_model_path}' …")
            hmm_model: HMM = utils.load_hmm_model(args.hmm_model_path, device=device)
            weights_tensor = utils.load_weights(args.weights_path, device=device)
            hmm_model.set_weights(weights_tensor)
            expectation_cache = hmm_model.compute_backward_expectation(T=args.max_len)
            hmm_processor = HmmGuidedLogitsProcessor(
                hmm_model=hmm_model,
                expectation_cache=expectation_cache,
                a=args.a,
                tokenizer=gen_tokenizer,
                decode_transform=not args.no_decode_transform,
                dump_eap_path=args.dump_eap_path,
            )
        elif args.hmm_variant == "hmm2":
            print(f"Loading second-order HMM (SOHMM) from '{args.hmm_model_path}' …")
            sohmm_model: SOHMM = utils.load_sohmm_model(args.hmm_model_path, device=device)
            weights_tensor = utils.load_weights(args.weights_path, device=device)
            sohmm_model.set_weights(weights_tensor)
            expectation_cache = sohmm_model.compute_backward_expectation(T=args.max_len)
            if args.no_decode_transform:
                print(
                    "Warning: --no_decode_transform is not supported for --hmm_variant hmm2 "
                    "and will be ignored (SOHMM kernel has no no-transform path).",
                    file=sys.stderr,
                )
            hmm_processor = SOHmmGuidedLogitsProcessor(
                hmm_model=sohmm_model,
                expectation_cache=expectation_cache,
                a=args.a,
                tokenizer=gen_tokenizer,
                dump_eap_path=args.dump_eap_path,
            )
        elif args.hmm_variant == "chmm":
            print(f"Loading clone-hidden HMM (CHMM) from '{args.hmm_model_path}' …")
            chmm_model: CHMM = utils.load_chmm_model(args.hmm_model_path, device=device)
            weights_tensor = utils.load_weights(args.weights_path, device=device)
            chmm_model.set_weights(weights_tensor)
            expectation_cache = chmm_model.compute_backward_expectation(T=args.max_len)
            if args.no_decode_transform:
                print(
                    "Warning: --no_decode_transform is not supported for --hmm_variant chmm "
                    "and will be ignored (CHMM kernel has no no-transform path).",
                    file=sys.stderr,
                )
            hmm_processor = CHMMGuidedLogitsProcessor(
                hmm_model=chmm_model,
                expectation_cache=expectation_cache,
                a=args.a,
                tokenizer=gen_tokenizer,
                dump_eap_path=args.dump_eap_path,
            )
        else:
            raise ValueError(f"Unknown hmm_variant: {args.hmm_variant!r}")
    else:
        print("Running in baseline mode (no HMM guidance)")

    #
    # LOAD ALL PROMPTS FROM INPUT FILE
    #
    if not os.path.exists(args.prompts_path):
        raise FileNotFoundError(f"Prompt file '{args.prompts_path}' not found.")
    
    # Load all prompts with their original indices
    prompts: List[Tuple[int, str]] = []
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompts.append((idx, json.loads(line)["prompt"]["text"]))

    print(f"Total prompts to process: {len(prompts)}")

    #
    # BATCH GENERATION LOOP
    #
    # Track whether output file exists for proper header writing
    file_exists = os.path.exists(output_path)

    # Process prompts in batches for memory efficiency
    for batch_start in tqdm(range(0, len(prompts), args.prompt_batch_size), desc="Generating"):
        # Extract current batch of prompts
        batch_info = prompts[batch_start : batch_start + args.prompt_batch_size]
        if not batch_info:
            continue

        prompt_texts = [txt for _, txt in batch_info]

        # Truncate prompts if they would exceed model's context window
        max_model_len = getattr(gen_model.config, "max_position_embeddings", 512)
        max_prompt_len = max(max_model_len - args.max_len - 10, 10)

        inputs = gen_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        prompt_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Configure HMM processor for this batch of prompts (if using HMM guidance)
        if hmm_processor:
            hmm_processor.configure_for_prompts(prompt_ids)

        # Initialize storage for all continuations for this batch
        batch_continuations: Dict[int, List[str]] = {idx: [] for idx, _ in batch_info}
        baseline_continuations: Dict[int, List[str]] = {idx: [] for idx, _ in batch_info} if args.baseline else None
        
        # Generate in smaller sub-batches if num_generations > generation_batch_size
        loops = math.ceil(args.num_generations / args.generation_batch_size)

        for loop_idx in range(loops):
            num_to_generate = min(
                args.generation_batch_size,
                args.num_generations - loop_idx * args.generation_batch_size,
            )
            if num_to_generate <= 0:
                break

            if hmm_processor and num_to_generate > 1:
                expanded_prompt_ids = prompt_ids.repeat_interleave(num_to_generate, dim=0)
                expanded_attention_mask = attention_mask.repeat_interleave(num_to_generate, dim=0)
                hmm_processor.configure_for_prompts(expanded_prompt_ids)
                input_ids_for_generation = prompt_ids
                attention_mask_for_generation = attention_mask
            else:
                input_ids_for_generation = prompt_ids
                attention_mask_for_generation = attention_mask

            logits_processors = LogitsProcessorList([hmm_processor]) if hmm_processor else LogitsProcessorList([])

            with torch.no_grad():
                gen_seqs = gen_model.generate(
                    input_ids=input_ids_for_generation,
                    attention_mask=attention_mask_for_generation,
                    logits_processor=logits_processors,
                    max_new_tokens=args.max_len,
                    num_return_sequences=num_to_generate,
                    do_sample=True,
                    top_p=0.9,
                    top_k=0,
                    temperature=1.0,
                    pad_token_id=gen_tokenizer.pad_token_id,
                    eos_token_id=gen_tokenizer.eos_token_id,
                )

            prompt_len = prompt_ids.shape[1]
            for b_idx in range(len(batch_info)):
                orig_idx = batch_info[b_idx][0]
                for k in range(num_to_generate):
                    seq_idx = b_idx * num_to_generate + k
                    cont_ids = gen_seqs[seq_idx][prompt_len:]
                    cont_text = gen_tokenizer.decode(
                        cont_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    batch_continuations[orig_idx].append(cont_text)

        if args.baseline and hmm_processor:
            for loop_idx in range(loops):
                num_to_generate = min(
                    args.generation_batch_size,
                    args.num_generations - loop_idx * args.generation_batch_size,
                )
                if num_to_generate <= 0:
                    break

                with torch.no_grad():
                    baseline_gen_seqs = gen_model.generate(
                        input_ids=input_ids_for_generation,
                        attention_mask=attention_mask_for_generation,
                        max_new_tokens=args.max_len,
                        num_return_sequences=num_to_generate,
                        do_sample=True,
                        top_p=0.9,
                        top_k=0,
                        temperature=1.0,
                        pad_token_id=gen_tokenizer.pad_token_id,
                        eos_token_id=gen_tokenizer.eos_token_id,
                    )

                for b_idx in range(len(batch_info)):
                    orig_idx = batch_info[b_idx][0]
                    for k in range(num_to_generate):
                        seq_idx = b_idx * num_to_generate + k
                        cont_ids = baseline_gen_seqs[seq_idx][prompt_len:]
                        cont_text = gen_tokenizer.decode(
                            cont_ids, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True
                        )
                        baseline_continuations[orig_idx].append(cont_text)

        # Prepare rows for CSV output
        batch_rows = []
        for orig_idx, prompt_text in batch_info:
            row: Dict = {"index": orig_idx, "prefix": prompt_text}
            
            if args.baseline and hmm_processor:
                for i, cont in enumerate(batch_continuations[orig_idx][: args.num_generations]):
                    row[f"trace_gen_{i + 1}"] = json.dumps({"continuation": cont})
                for i, cont in enumerate(baseline_continuations[orig_idx][: args.num_generations]):
                    row[f"baseline_gen_{i + 1}"] = json.dumps({"continuation": cont})
            else:
                for i, cont in enumerate(batch_continuations[orig_idx][: args.num_generations]):
                    mode_prefix = "baseline" if not hmm_processor else "trace"
                    row[f"{mode_prefix}_gen_{i + 1}"] = json.dumps({"continuation": cont})
            batch_rows.append(row)

        pd.DataFrame(batch_rows).to_csv(
            output_path,
            mode="a",
            header=not file_exists,
            index=False,
        )
        file_exists = True
        print(f"Saved {len(batch_rows)} rows → {output_path}")

    print("Generation complete ✔ – results in", output_path)


if __name__ == "__main__":
    main()
