"""CPU-only smoke tests for the metric helpers."""

import time

import torch


def test_throughput_returns_positive():
    """measure_throughput should run and return a positive number."""
    from src.evaluate import measure_throughput

    # mock the bare minimum HF surface so we don't need to load a real model
    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            time.sleep(0.001)
            return torch.zeros(1, kwargs.get("max_new_tokens", 1), dtype=torch.long)

    class FakeTokenizer:
        def __call__(self, _text, return_tensors="pt"):
            class Out:
                def to(self, _device):
                    return self
            o = Out()
            o.input_ids = torch.zeros(1, 4, dtype=torch.long)
            return o

    tps = measure_throughput(
        FakeModel(), FakeTokenizer(),
        prompt="hi", new_tokens=4, n_warmup=1,
    )
    assert tps > 0


def test_peak_memory_handles_cpu():
    from src.evaluate import measure_peak_memory_gb, reset_peak_memory

    reset_peak_memory()
    val = measure_peak_memory_gb()
    if not torch.cuda.is_available():
        assert val == 0.0
    else:
        assert val >= 0.0
