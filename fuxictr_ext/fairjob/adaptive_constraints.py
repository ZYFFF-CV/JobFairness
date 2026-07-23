"""Graph-risk aggregation and optional adaptive dual state for Stage2."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


PROXY_CHANCE_BCE = math.log(2.0)


def proxy_advantages(
    losses: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert balanced BCE to chance-relative proxy prediction advantage."""

    return {
        name: torch.clamp(
            torch.as_tensor(
                PROXY_CHANCE_BCE, dtype=value.dtype, device=value.device
            )
            - value,
            min=0.0,
        )
        for name, value in losses.items()
    }


class GraphRiskAggregator(nn.Module):
    """Aggregate adversary BCE while emphasizing the easiest proxy target.

    The encoder receives reversed gradients from every adversary input. The
    smooth-max proxy-advantage objective is therefore implemented as a
    normalized soft-min of BCE losses. Minimizing it trains adversaries toward
    the currently easiest node, while gradient reversal makes the encoder raise
    that worst-node BCE.
    """

    MODES = {"smooth_max_proxy_advantage", "mean_bce", "cvar_low_bce"}

    def __init__(
        self,
        mode: str = "smooth_max_proxy_advantage",
        temperature: float = 0.05,
        cvar_fraction: float = 0.25,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown graph-risk mode: {mode}")
        if temperature <= 0:
            raise ValueError("Graph-risk temperature must be positive.")
        if not 0 < cvar_fraction <= 1:
            raise ValueError("CVaR fraction must be in (0, 1].")
        self.mode = mode
        self.temperature = float(temperature)
        self.cvar_fraction = float(cvar_fraction)

    def forward(
        self, losses: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        if not losses:
            raise ValueError("Graph-risk aggregation requires node losses.")
        names = list(losses)
        values = torch.stack([losses[name] for name in names])
        if self.mode == "mean_bce":
            objective = values.mean()
        elif self.mode == "smooth_max_proxy_advantage":
            count = torch.as_tensor(
                len(values), dtype=values.dtype, device=values.device
            )
            objective = (
                -self.temperature
                * torch.logsumexp(-values / self.temperature, dim=0)
                + self.temperature * torch.log(count)
            )
        else:
            count = max(1, int(math.ceil(len(values) * self.cvar_fraction)))
            objective = torch.topk(values, k=count, largest=False).values.mean()
        detached = values.detach()
        worst_index = int(torch.argmin(detached).cpu())
        advantages = proxy_advantages(losses)
        telemetry = {
            "mode": self.mode,
            "worst_node": names[worst_index],
            "node_bce": {
                name: float(losses[name].detach().cpu()) for name in names
            },
            "node_proxy_advantage": {
                name: float(value.detach().cpu())
                for name, value in advantages.items()
            },
            "objective": float(objective.detach().cpu()),
        }
        return objective, telemetry


class AdaptiveDualController(nn.Module):
    """Track non-negative multipliers for validation-level risk constraints."""

    def __init__(
        self,
        names: list[str],
        thresholds: Mapping[str, float],
        learning_rate: float = 0.01,
        maximum: float = 10.0,
    ):
        super().__init__()
        if not names or len(set(names)) != len(names):
            raise ValueError("Dual constraint names must be unique and non-empty.")
        if set(names) != set(thresholds):
            raise ValueError("Every dual constraint requires one threshold.")
        if learning_rate <= 0 or maximum <= 0:
            raise ValueError("Dual update parameters must be positive.")
        self.names = list(names)
        self.thresholds = {
            name: float(thresholds[name]) for name in self.names
        }
        self.learning_rate = float(learning_rate)
        self.maximum = float(maximum)
        self.register_buffer("multipliers", torch.zeros(len(names)))

    @torch.no_grad()
    def update(self, risks: Mapping[str, float]) -> dict[str, float]:
        """Apply one projected dual-ascent update from frozen-scope risks."""

        if set(risks) != set(self.names):
            raise ValueError("Dual update risks do not match frozen constraints.")
        violations = torch.tensor(
            [
                float(risks[name]) - self.thresholds[name]
                for name in self.names
            ],
            dtype=self.multipliers.dtype,
            device=self.multipliers.device,
        )
        self.multipliers.add_(self.learning_rate * violations)
        self.multipliers.clamp_(0.0, self.maximum)
        return {
            name: float(self.multipliers[index].cpu())
            for index, name in enumerate(self.names)
        }
