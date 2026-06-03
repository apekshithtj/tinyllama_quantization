"""
Activation analysis with forward hooks.

Captures intermediate activations from a few interesting layers at
both FP16 and INT4, and plots their distributions. The interesting
thing to look for: heavy tails (large outliers) in a layer's
activations usually correlate with that layer being a bad fit for
aggressive INT4 quantization.

Hooks are only used here. Not during the main metric runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.activations import collect_activations
from src.loader import load_fp16_model, load_int4_model, load_tokenizer
from src.plotting import plot_activation_histograms

# o_proj is the output of attention before the residual add.
# down_proj is the same role for MLP. Both are known activation-heavy
# in transformer LLMs (see LLM.int8(), SmoothQuant, AWQ papers).
# Sampling early/middle/late layers so we can see how things shift with depth.
TARGETS = [
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.10.self_attn.o_proj",
    "model.layers.10.mlp.down_proj",
    "model.layers.20.self_attn.o_proj",
    "model.layers.20.mlp.down_proj",
]


def main():
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")

    tok = load_tokenizer()

    print("\n=== FP16 activations ===")
    fp16 = load_fp16_model()
    fp16.eval()
    fp16_acts = collect_activations(fp16, tok, TARGETS)
    del fp16
    torch.cuda.empty_cache()

    print("\n=== INT4 activations ===")
    int4 = load_int4_model()
    int4.eval()
    int4_acts = collect_activations(int4, tok, TARGETS)
    del int4
    torch.cuda.empty_cache()

    # tag the keys so the subplot titles say which precision they're from
    plot_activation_histograms(
        {f"{k} [FP16]": v for k, v in fp16_acts.items()},
        filename="activations_fp16.png",
    )
    plot_activation_histograms(
        {f"{k} [INT4]": v for k, v in int4_acts.items()},
        filename="activations_int4.png",
    )

    print("\nstats:")
    print(f"{'layer':45s} {'fp16_max':>10s} {'int4_max':>10s} {'fp16_std':>10s} {'int4_std':>10s}")
    for n in TARGETS:
        a = fp16_acts[n].float()
        b = int4_acts[n].float()
        print(f"{n:45s} {a.abs().max().item():10.3f} {b.abs().max().item():10.3f} "
              f"{a.std().item():10.4f} {b.std().item():10.4f}")


if __name__ == "__main__":
    main()
