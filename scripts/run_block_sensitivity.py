"""
Per-block INT4 sweep: attention vs MLP inside each layer.
About 2x the runtime of run_layer_sensitivity.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.evaluate import evaluate_perplexity
from src.loader import load_fp16_model, load_tokenizer
from src.plotting import plot_block_sensitivity
from src.sensitivity import block_by_block


def main():
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")

    tok = load_tokenizer()
    fp16 = load_fp16_model()
    fp16.eval()

    print("\n=== FP16 baseline ===")
    baseline = evaluate_perplexity(fp16, tok, n_chunks=8)
    print(f"baseline ppl (8 chunks): {baseline:.4f}")

    print("\n=== per-block sweep ===")
    results = block_by_block(fp16, tok, n_chunks=8)

    os.makedirs("results", exist_ok=True)
    with open("results/block_sensitivity.json", "w") as f:
        json.dump({"baseline_ppl": baseline, "per_block_ppl": results}, f, indent=2)
    print("\nwrote results/block_sensitivity.json")

    avg_attn = sum(v["attention"] for v in results.values()) / len(results)
    avg_mlp = sum(v["mlp"] for v in results.values()) / len(results)
    print(f"\navg ppl, attention INT4: {avg_attn:.4f}")
    print(f"avg ppl, MLP INT4:       {avg_mlp:.4f}")
    print(f"baseline:                {baseline:.4f}")
    print("attention more sensitive" if avg_attn > avg_mlp else "mlp more sensitive")

    plot_block_sensitivity(results, baseline)


if __name__ == "__main__":
    main()
