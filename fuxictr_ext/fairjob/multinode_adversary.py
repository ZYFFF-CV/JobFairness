"""Train-only proxy adversaries for Stage2 graph nodes and joint paths."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(inputs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Leave forward values unchanged and reverse encoder gradients."""

    return _GradientReverse.apply(inputs, scale)


def balanced_binary_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Average group-conditional BCE so proxy imbalance cannot dominate."""

    logits = logits.reshape(-1)
    target = target.float().reshape(-1)
    if len(logits) != len(target):
        raise ValueError("Adversary logits and targets have different rows.")
    group_losses = []
    for group in (0.0, 1.0):
        selected = target == group
        if selected.any():
            group_losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[selected], target[selected]
                )
            )
    if len(group_losses) < 2:
        # A one-group minibatch has no balanced proxy contrast. Returning a
        # connected zero avoids a biased update while preserving autograd.
        return logits.sum() * 0.0
    return torch.stack(group_losses).mean()


def _adversary(input_dim: int, hidden_units: tuple[int, ...]) -> nn.Sequential:
    layers = []
    current = int(input_dim)
    for hidden in hidden_units:
        layers.extend((nn.Linear(current, int(hidden)), nn.ReLU()))
        current = int(hidden)
    layers.append(nn.Linear(current, 1))
    module = nn.Sequential(*layers)
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return module


class MultiNodeAdversary(nn.Module):
    """Apply matched-capacity adversaries to named graph representations."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        hidden_units: tuple[int, ...] = (64,),
        gradient_reversal_scale: float = 1.0,
    ):
        super().__init__()
        if not input_dims:
            raise ValueError("At least one adversary input is required.")
        if any(int(value) < 1 for value in input_dims.values()):
            raise ValueError("Adversary input dimensions must be positive.")
        self.input_dims = {
            str(name): int(value) for name, value in input_dims.items()
        }
        self.gradient_reversal_scale = float(gradient_reversal_scale)
        self.adversaries = nn.ModuleDict(
            {
                name: _adversary(dimension, hidden_units)
                for name, dimension in self.input_dims.items()
            }
        )

    def forward(
        self, representations: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return one proxy logit per preregistered graph target."""

        missing = sorted(set(self.input_dims).difference(representations))
        if missing:
            raise ValueError(f"Missing adversary representations: {missing}")
        output = {}
        for name, dimension in self.input_dims.items():
            value = representations[name]
            if value.ndim != 2 or value.shape[1] != dimension:
                raise ValueError(
                    f"Adversary node {name} expected [N,{dimension}], "
                    f"received {list(value.shape)}."
                )
            output[name] = self.adversaries[name](
                gradient_reverse(value, self.gradient_reversal_scale)
            )
        return output

    @staticmethod
    def losses(
        logits: Mapping[str, torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute matched balanced BCE for every node adversary."""

        return {
            name: balanced_binary_loss(value, target)
            for name, value in logits.items()
        }
