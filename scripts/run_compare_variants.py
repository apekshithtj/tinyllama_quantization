"""
Headline experiment: FP16 vs INT8 vs INT4. Run this first.

Writes results/variants.json and results/figures/pareto.png.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.evaluate import summarize_run
from src.loader import load_fp16_model, load_int4_model, load_int8_model, load_tokenizer
from src.plotting import plot_pareto


def main():
    if not torch.cuda.is_available():
        sys.exit("needs CUDA, bitsandbytes won't run on CPU")

    tok = load_tokenizer()
    rows = []

    variants = [
        ("FP16", load_fp16_model),
        ("INT8 (bnb)", load_int8_model),
        ("INT4 (bnb nf4)", load_int4_model),
    ]

    for label, loader in variants:
        print(f"\n=== {label} ===")
        m = loader()
        m.eval()
        rows.append(summarize_run(m, tok, label, n_chunks=20))
        # without this cleanup the second/third load OOMs
        del m
        torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/variants.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote results/variants.json")

    plot_pareto(rows)


if __name__ == "__main__":
    main()
