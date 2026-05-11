from __future__ import annotations

from typing import Callable

import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layer_names: list[str] = []

    def step(self, loss_fn: Callable[[], float]) -> float:
        return float(loss_fn())