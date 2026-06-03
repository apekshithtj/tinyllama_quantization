"""
Swap individual nn.Linear layers for bitsandbytes Linear4bit.

The whole sensitivity analysis depends on being able to quantize one
layer (or one block) at a time and leave the rest untouched. GPTQ/AWQ
are whole-model algorithms so they're awkward for this; bitsandbytes
lets us do it per-module.
"""

import copy

import bitsandbytes as bnb
import torch
import torch.nn as nn


def _get_parent_and_attr(root, dotted):
    # walk a dotted path like 'model.layers.5.self_attn.q_proj' and
    # return (parent, last_attr) so setattr can replace the leaf.
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def replace_linear_with_int4(model, dotted_name):
    """
    Replace a single nn.Linear with a bnb Linear4bit. In place.

    The order of operations here matters and isn't obvious:
    build the Linear4bit, attach a Params4bit wrapping the FP16
    weights, THEN move to CUDA. The move is what triggers
    quantization. If you do .to() before assigning Params4bit,
    some bnb versions silently skip the quantization step.
    """
    parent, attr = _get_parent_and_attr(model, dotted_name)
    old = getattr(parent, attr)

    if not isinstance(old, nn.Linear):
        raise TypeError(f"{dotted_name} is a {type(old).__name__}, not nn.Linear")

    device = old.weight.device
    has_bias = old.bias is not None

    new = bnb.nn.Linear4bit(
        old.in_features,
        old.out_features,
        bias=has_bias,
        quant_type="nf4",
        compute_dtype=torch.float16,
    )
    new.weight = bnb.nn.Params4bit(
        data=old.weight.data.contiguous(),
        requires_grad=False,
        quant_type="nf4",
    )
    if has_bias:
        new.bias = nn.Parameter(old.bias.data.clone())
    new = new.to(device)  # this is the call that actually quantizes

    setattr(parent, attr, new)


def list_linear_modules(model, skip=("lm_head",)):
    """Every nn.Linear in the model, by dotted name. Skips lm_head by default."""
    skip = set(skip)
    return [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not any(s in n for s in skip)
    ]


def list_decoder_layers(model):
    """Names like 'model.layers.0', 'model.layers.1', ..."""
    # match exactly two dots so 'model.layers.0' counts but
    # 'model.layers.0.self_attn' (three dots) does not.
    return [
        n for n, _ in model.named_modules()
        if n.startswith("model.layers.") and n.count(".") == 2
    ]


def linears_inside(model, prefix):
    parent = model.get_submodule(prefix)
    out = []
    for sub_name, m in parent.named_modules():
        if isinstance(m, nn.Linear):
            out.append(f"{prefix}.{sub_name}" if sub_name else prefix)
    return out


def linears_in_attention(model, layer_prefix):
    """q, k, v, o projections inside a decoder layer's self-attention."""
    return linears_inside(model, f"{layer_prefix}.self_attn")


def linears_in_mlp(model, layer_prefix):
    """gate, up, down projections inside a decoder layer's MLP."""
    return linears_inside(model, f"{layer_prefix}.mlp")


def clone_fp16_model(model):
    # deepcopy needs room for two copies on GPU. Fine for TinyLlama on T4
    # (~4.5 GB total), don't try this with 7B on the same card.
    return copy.deepcopy(model)
