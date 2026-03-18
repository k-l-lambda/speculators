#!/usr/bin/env python3
"""Compare vLLM draft logits dump with offline eval logits for the same sample.

Usage:
    # 1. Run vLLM with VLLM_DUMP_DRAFT_LOGITS=/tmp/draft_dump
    # 2. Send data_100 prompt to vLLM
    # 3. Run offline eval to get eval logits
    # 4. python tests/compare_draft_logits.py \
    #        --vllm-dump /tmp/draft_dump \
    #        --eval-logits /tmp/data100_eval_logits.pt

    Compares per-position logits, predictions, and identifies divergence sources.
"""
import argparse
import glob
import os
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dump", required=True, help="Directory with draft_step_*.pt files")
    parser.add_argument("--eval-logits", required=True, help="Path to eval logits .pt file")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps to compare")
    args = parser.parse_args()

    # Load eval data
    eval_data = torch.load(args.eval_logits, map_location="cpu", weights_only=True)
    eval_logits = eval_data["logits"]  # [num_resp_positions, vocab_size]
    eval_preds = eval_data["preds"]
    eval_targets = eval_data["targets"]
    eval_x_ids = eval_data["x_ids"]
    eval_positions = eval_data["positions"]

    print(f"Eval: {len(eval_positions)} response positions, logits shape {eval_logits.shape}")

    # Load vLLM dump
    dump_files = sorted(glob.glob(os.path.join(args.vllm_dump, "draft_step_*.pt")))
    print(f"vLLM dump: {len(dump_files)} steps")

    n = min(len(dump_files), len(eval_positions), args.max_steps)
    print(f"\nComparing first {n} positions:")
    print("%4s %7s %8s %8s %8s %8s %8s %5s %8s %8s" % ("Step", "EvalPos", "EvalInp", "EvalPred", "EvalTgt", "vLLMInp", "vLLMPred", "Match", "CosSim", "MaxDiff"))

    match_count = 0
    cos_sims = []
    for i in range(n):
        # Eval side
        e_pos = eval_positions[i]
        e_inp = eval_x_ids[i].item()
        e_pred = eval_preds[i].item()
        e_tgt = eval_targets[i].item()
        e_logits = eval_logits[i]

        # vLLM side
        v_data = torch.load(dump_files[i], map_location="cpu", weights_only=True)
        v_logits = v_data["logits"].float()
        v_draft_id = v_data["draft_ids"]
        v_inp_id = v_data["input_ids"]

        # Handle batch dimension
        if v_logits.dim() == 2:
            # Multiple tokens in batch, take first
            v_logits_pos = v_logits[0]
            v_pred = v_draft_id[0].item() if v_draft_id.dim() > 0 else v_draft_id.item()
            v_inp = v_inp_id[0].item() if v_inp_id.dim() > 0 else v_inp_id.item()
        else:
            v_logits_pos = v_logits
            v_pred = v_draft_id.item()
            v_inp = v_inp_id.item()

        # Compare
        pred_match = "Y" if e_pred == v_pred else "N"
        if e_pred == v_pred:
            match_count += 1

        # Cosine similarity of logit vectors
        cos = torch.nn.functional.cosine_similarity(e_logits.unsqueeze(0), v_logits_pos.unsqueeze(0)).item()
        cos_sims.append(cos)

        # Max absolute difference
        max_diff = (e_logits - v_logits_pos).abs().max().item()

        print(f"{i:4d} {e_pos:7d} {e_inp:8d} {e_pred:8d} {e_tgt:8d} {v_inp:8d} {v_pred:8d} {pred_match:>5} {cos:8.4f} {max_diff:8.2f}")

    print(f"\nPrediction match: {match_count}/{n} = {match_count/n:.4f}")
    print(f"Cosine similarity: mean={np.mean(cos_sims):.4f}, min={np.min(cos_sims):.4f}, max={np.max(cos_sims):.4f}")


if __name__ == "__main__":
    main()
