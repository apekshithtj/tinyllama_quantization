"""
Forward hooks for grabbing intermediate activations.

Only used by the analysis script, not during the metric runs.
Hooks add overhead and if you forget to remove them they keep
firing on every subsequent forward pass.
"""

import torch
import torch.nn as nn


class ActivationCatcher:
    """
    Context manager that hooks a list of named modules and stashes
    their outputs into self.activations.

    Usage:
        with ActivationCatcher(model, ["model.layers.0.mlp.down_proj"]) as cap:
            model(**inputs)
        cap.activations  # {name: tensor}
    """

    def __init__(self, model, module_names):
        self.model = model
        self.module_names = module_names
        self.activations = {}
        self._handles = []

    def _make_hook(self, name):
        # closure over `name` so each hook knows which key to write to.
        def hook(_module, _inputs, output):
            # attention modules return (output, attn_weights); we only want the first.
            if isinstance(output, tuple):
                output = output[0]
            # move to CPU and float32 so plotting works the same regardless
            # of whether the model was fp16 or int4.
            self.activations[name] = output.detach().to("cpu", torch.float32)
        return hook

    def __enter__(self):
        for name in self.module_names:
            mod = self.model.get_submodule(name)
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def collect_activations(model, tokenizer, module_names,
                        prompt="The capital of France is Paris and the city is known for"):
    """One forward pass with hooks, return what got captured."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with ActivationCatcher(model, module_names) as cap:
        with torch.no_grad():
            _ = model(**inputs)
    return cap.activations
