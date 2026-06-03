"""
Tests for the quantization helpers. Needs a GPU.

The main one is test_int4_output_close_to_fp16 which uses MSE between
the FP16 reference and the INT4 output as a quick sanity check that
quantization actually ran and didn't break anything structural.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bitsandbytes needs CUDA",
)


def _linear(in_f=128, out_f=256):
    # seeded so two calls produce the same weights
    torch.manual_seed(0)
    return nn.Linear(in_f, out_f, bias=False).half().cuda()


def _mse(a, b):
    return torch.nn.functional.mse_loss(a.float(), b.float()).item()


def test_only_target_layer_changes():
    """quantizing 'a' must not touch 'b'."""
    from src.quantize import replace_linear_with_int4

    class TwoLinears(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _linear()
            self.b = _linear()

    model = TwoLinears().cuda()
    b_before = model.b.weight.detach().clone()

    replace_linear_with_int4(model, "a")

    assert type(model.a).__name__ == "Linear4bit"
    assert isinstance(model.b, nn.Linear)
    assert torch.equal(model.b.weight, b_before)


def test_int4_close_to_fp16():
    """
    Same input through the FP16 and INT4 versions of the same weights
    should give very similar outputs. MSE under 0.05 on this scale.
    """
    from src.quantize import replace_linear_with_int4

    class OneLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _linear(256, 256)

    torch.manual_seed(1)
    x = torch.randn(4, 16, 256, dtype=torch.float16, device="cuda")

    fp16 = OneLinear().cuda()
    with torch.no_grad():
        ref = fp16.proj(x)

    int4 = OneLinear().cuda()
    int4.proj.weight.data.copy_(fp16.proj.weight.data)  # belt and braces
    replace_linear_with_int4(int4, "proj")
    with torch.no_grad():
        out = int4.proj(x)

    mse = _mse(ref, out)
    # observed ~1e-3 to 5e-3 on random weights; 0.05 catches structural bugs.
    assert mse < 0.05, f"mse={mse:.4f}, something's off"


def test_int4_output_finite():
    """no NaN, no Inf, no values that have blown up to thousands."""
    from src.quantize import replace_linear_with_int4

    class OneLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _linear()

    m = OneLinear().cuda()
    replace_linear_with_int4(m, "proj")

    x = torch.randn(2, 10, 128, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        y = m.proj(x)

    assert torch.isfinite(y).all()
    assert y.abs().max().item() < 1e3


def test_list_decoder_layers():
    """walks a fake llama-shaped tree, no real model needed."""
    from src.quantize import list_decoder_layers

    class FakeLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

    out = list_decoder_layers(FakeLlama())
    assert out == ["model.layers.0", "model.layers.1", "model.layers.2"]
