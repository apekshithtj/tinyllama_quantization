"""
Sensitivity sweeps. Layer-by-layer and block-by-block.

The pattern is always the same: clone the FP16 baseline, swap a chunk
of it to INT4, measure ppl, drop the clone. Repeat.
"""

import torch

from .evaluate import evaluate_perplexity
from .quantize import (
    clone_fp16_model,
    linears_in_attention,
    linears_in_mlp,
    linears_inside,
    list_decoder_layers,
    replace_linear_with_int4,
)


def _free(model):
    # called between iterations so the cached allocator doesn't pile up
    # deepcopies and OOM us after a few layers.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def layer_by_layer(fp16_model, tokenizer, n_chunks=8):
    """
    For each decoder layer, quantize the whole layer to INT4 and
    measure ppl. Returns {layer_name: ppl}.

    n_chunks is small here on purpose. We run this 22 times and a
    full ppl sweep per iteration would take ages. 8 chunks ranks
    layers reliably even if the absolute numbers wobble.
    """
    layer_names = list_decoder_layers(fp16_model)
    print(f"found {len(layer_names)} decoder layers")
    results = {}

    for i, name in enumerate(layer_names):
        print(f"\n[{i + 1}/{len(layer_names)}] quantizing {name}")
        m = clone_fp16_model(fp16_model)
        for t in linears_inside(m, name):
            replace_linear_with_int4(m, t)
        ppl = evaluate_perplexity(m, tokenizer, n_chunks=n_chunks)
        results[name] = ppl
        print(f"   ppl = {ppl:.4f}")
        _free(m)

    return results


def block_by_block(fp16_model, tokenizer, n_chunks=8):
    """
    For each decoder layer, run two experiments: attention-only INT4
    and MLP-only INT4. Returns nested dict.
    """
    layer_names = list_decoder_layers(fp16_model)
    results = {}

    for i, name in enumerate(layer_names):
        results[name] = {}
        print(f"\n[{i + 1}/{len(layer_names)}] block sweep for {name}")

        # attention block
        m = clone_fp16_model(fp16_model)
        for t in linears_in_attention(m, name):
            replace_linear_with_int4(m, t)
        ppl_attn = evaluate_perplexity(m, tokenizer, n_chunks=n_chunks)
        results[name]["attention"] = ppl_attn
        print(f"   attention -> ppl = {ppl_attn:.4f}")
        _free(m)

        # mlp block
        m = clone_fp16_model(fp16_model)
        for t in linears_in_mlp(m, name):
            replace_linear_with_int4(m, t)
        ppl_mlp = evaluate_perplexity(m, tokenizer, n_chunks=n_chunks)
        results[name]["mlp"] = ppl_mlp
        print(f"   mlp       -> ppl = {ppl_mlp:.4f}")
        _free(m)

    return results
