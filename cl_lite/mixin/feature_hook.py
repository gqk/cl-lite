# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Any, Union, Callable

import torch.nn as nn


class FeatureHook:
    value: Any = None

    def __init__(
        self,
        name,
        module: nn.Module,
        transform: Callable = None,
        inout: str = "out",
    ):
        assert inout in ["in", "in[0]", "out"]

        self.name = name
        self.value = None
        self.transform = transform
        self.inout = inout
        self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        value = {"in": input, "in[0]": input[0], "out": output}[self.inout]
        if self.transform is not None:
            value = self.transform(value)
        self.value = value

    def close(self):
        self.handle.remove()


def get_submodule(module: nn.Module, path: str):
    for attr in path.split("."):
        module = getattr(module, attr, None)
        assert isinstance(module, nn.Module)
    return module


class FeatureHookMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self._feature_hooks = OrderedDict()

    def clear_feature_hooks(self, reset_value: bool = False):
        for hook in self._feature_hooks.values():
            if reset_value:
                hook.value = None
            else:
                hook.close()
        if not reset_value:
            self._feature_hooks = OrderedDict()

    def unregister_feature_hook(self, name: str):
        hook = self._feature_hooks.pop(name, None)
        if hook is not None:
            hook.close()

    def register_feature_hook(
        self,
        name: str,
        submodule: Union[str, nn.Module],
        transform: Callable = None,
        inout: str = "out",
    ):
        if isinstance(submodule, str):
            submodule = get_submodule(self, submodule)

        self.unregister_feature_hook(name)

        args = [name, submodule, transform, inout]
        self._feature_hooks[name] = FeatureHook(*args)

    def current_feature(self, name: str):
        return self._feature_hooks[name].value
