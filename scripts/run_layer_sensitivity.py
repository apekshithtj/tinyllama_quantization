"""
Per-layer INT4 sensitivity sweep.

For each decoder layer, quantize only that layer to INT4 and measure
ppl. ~30-40 min on a T4.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.evaluate import evaluate_perplexity
from src.loader import load_fp16_model, load_tokenizer
from src.plotting import plot_layer_sensitivity
from src.sensitivity import layer_by_layer


def main():
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")

    tok = load_tokenizer()
    fp16 = load_fp16_model()
    fp16.eval()

    # measure baseline with the same n_chunks as the sweep, otherwise the
    # reference line in the plot is misleading.
    print("\n=== FP16 baseline ===")
    baseline = evaluate_perplexity(fp16, tok, n_chunks=8)
    print(f"baseline ppl (8 chunks): {baseline:.4f}")

    print("\n=== per-layer sweep ===")
    results = layer_by_layer(fp16, tok, n_chunks=8)

    os.makedirs("results", exist_ok=True)
    with open("results/layer_sensitivity.json", "w") as f:
        json.dump({"baseline_ppl": baseline, "per_layer_ppl": results}, f, indent=2)
    print("\nwrote results/layer_sensitivity.json")

    ranked = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print("\ntop 5 most sensitive:")
    for name, ppl in ranked[:5]:
        print(f"  {name:25s}  ppl={ppl:.4f}  (+{ppl - baseline:.4f})")

    plot_layer_sensitivity(results, baseline)


if __name__ == "__main__":
    main()
