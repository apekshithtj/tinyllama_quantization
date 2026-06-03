"""matplotlib helpers. all plots land in results/figures/."""

import os

import matplotlib

# Agg backend so this works headless (Colab, ssh, CI).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join("results", "figures")


def _save(fig, filename):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def plot_pareto(rows, filename="pareto.png"):
    """ppl vs peak memory, one point per variant."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for row in rows:
        ax.scatter(row["peak_memory_gb"], row["perplexity"], s=120)
        ax.annotate(
            row["label"],
            (row["peak_memory_gb"], row["perplexity"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Peak GPU memory (GB)")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_title("Accuracy vs memory")
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_layer_sensitivity(results, baseline_ppl, filename="layer_sensitivity.png"):
    # sort by numeric layer index, not by string (so layer 10 doesn't come before layer 2)
    items = sorted(results.items(), key=lambda kv: int(kv[0].split(".")[-1]))
    names = [n.split(".")[-1] for n, _ in items]
    ppls = [p for _, p in items]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, ppls)
    margin = max(ppls) - min(min(ppls), baseline_ppl)
    ax.set_ylim(min(min(ppls), baseline_ppl) - margin * 0.3,
                max(ppls) + margin * 0.3)
    ax.axhline(baseline_ppl, linestyle="--", color="black",
               label=f"FP16 baseline = {baseline_ppl:.2f}")
    ax.set_xlabel("Decoder layer index")
    ax.set_ylabel("Perplexity (this layer at INT4)")
    ax.set_title("Per-layer sensitivity to INT4")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, filename)


def plot_block_sensitivity(results, baseline_ppl, filename="block_sensitivity.png"):
    items = sorted(results.items(), key=lambda kv: int(kv[0].split(".")[-1]))
    idx = np.arange(len(items))
    attn_ppls = [v["attention"] for _, v in items]
    mlp_ppls = [v["mlp"] for _, v in items]
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(idx - w / 2, attn_ppls, w, label="Attention INT4")
    ax.bar(idx + w / 2, mlp_ppls, w, label="MLP INT4")
    all_vals = attn_ppls + mlp_ppls + [baseline_ppl]
    margin = max(all_vals) - min(all_vals)
    ax.set_ylim(min(all_vals) - margin * 0.3, max(all_vals) + margin * 0.3)
    ax.axhline(baseline_ppl, linestyle="--", color="black",
               label=f"FP16 baseline = {baseline_ppl:.2f}")
    ax.set_xticks(idx)
    ax.set_xticklabels([n.split(".")[-1] for n, _ in items])
    ax.set_xlabel("Decoder layer index")
    ax.set_ylabel("Perplexity")
    ax.set_title("Attention vs MLP per layer")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, filename)


def plot_activation_histograms(activations, filename="activations.png", bins=120):
    """
    one subplot per layer. log y so tails are visible (linear scale
    makes the bulk near zero swamp everything else).
    """
    names = list(activations.keys())
    n = len(names)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        values = activations[name].flatten().numpy()
        ax.hist(values, bins=bins)
        ax.set_title(name, fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Activation distributions (log y)")
    fig.tight_layout()
    return _save(fig, filename)
