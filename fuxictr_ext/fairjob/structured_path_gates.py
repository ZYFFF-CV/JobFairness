"""Budget-matched gates over named DCNv2 blocks and branches."""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn


class StructuredPathGates(nn.Module):
    """Allocate one fixed suppression budget across architecture-level units."""

    MODES = {"identity", "learned", "frozen_random"}

    def __init__(
        self,
        names: list[str],
        mode: str = "learned",
        suppression_budget_per_gate: float = 0.1,
        minimum_keep: float = 0.5,
        temperature: float = 1.0,
        seed: int = 2019,
    ):
        super().__init__()
        if not names or len(set(names)) != len(names):
            raise ValueError("Structured gate names must be unique and non-empty.")
        if mode not in self.MODES:
            raise ValueError(f"Unknown structured gate mode: {mode}")
        if not 0 <= suppression_budget_per_gate < 1:
            raise ValueError("Gate suppression budget must be in [0, 1).")
        if not 0 < minimum_keep <= 1:
            raise ValueError("minimum_keep must be in (0, 1].")
        if temperature <= 0:
            raise ValueError("Gate temperature must be positive.")
        total_budget = suppression_budget_per_gate * len(names)
        maximum_total = (1.0 - minimum_keep) * len(names)
        if total_budget > maximum_total:
            raise ValueError(
                "Total suppression budget exceeds the minimum-keep capacity."
            )
        self.names = list(names)
        self.mode = mode
        self.suppression_budget_per_gate = float(
            suppression_budget_per_gate
        )
        self.minimum_keep = float(minimum_keep)
        self.temperature = float(temperature)
        self.seed = int(seed)
        if mode == "learned":
            self.allocation_logits = nn.Parameter(torch.zeros(len(names)))
        elif mode == "frozen_random":
            generator = torch.Generator().manual_seed(seed)
            self.register_buffer(
                "allocation_logits",
                torch.randn(len(names), generator=generator),
            )
        else:
            self.register_buffer(
                "allocation_logits", torch.zeros(len(names))
            )

    def keep_values(self) -> dict[str, torch.Tensor]:
        """Return scalar keep values under the frozen aggregate budget."""

        if self.mode == "identity" or self.suppression_budget_per_gate == 0:
            values = torch.ones_like(self.allocation_logits)
        else:
            learned_weights = torch.softmax(
                self.allocation_logits / self.temperature, dim=0
            )
            total_budget = self.suppression_budget_per_gate * len(self.names)
            # Mix with a uniform allocation when necessary so every feasible
            # parameter value preserves both the exact total budget and the
            # minimum-keep bound. Clamping would silently spend less budget and
            # make learned and random controls incomparable.
            uniform = torch.full_like(
                learned_weights, 1.0 / len(self.names)
            )
            maximum_share = (1.0 - self.minimum_keep) / total_budget
            if maximum_share >= 1.0:
                weights = learned_weights
            else:
                maximum_mix = (
                    maximum_share - 1.0 / len(self.names)
                ) / (1.0 - 1.0 / len(self.names))
                mix = min(1.0, max(0.0, maximum_mix))
                weights = uniform + mix * (learned_weights - uniform)
            suppression = total_budget * weights
            values = 1.0 - suppression
        return {
            name: values[index] for index, name in enumerate(self.names)
        }

    def apply(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Apply one named scalar gate without changing tensor dimensions."""

        if name not in self.names:
            raise KeyError(f"Unknown structured gate: {name}")
        return value * self.keep_values()[name]

    def telemetry(self) -> dict:
        """Return detached keep/suppression values for live training logs."""

        keep = self.keep_values()
        values = {
            name: float(value.detach().cpu()) for name, value in keep.items()
        }
        return {
            "mode": self.mode,
            "keep": values,
            "suppression": {
                name: 1.0 - value for name, value in values.items()
            },
            "configured_total_suppression": (
                self.suppression_budget_per_gate * len(self.names)
            ),
            "nominal_total_suppression": (
                0.0
                if self.mode == "identity"
                else self.suppression_budget_per_gate * len(self.names)
            ),
            "actual_total_suppression": sum(1.0 - value for value in values.values()),
            "minimum_keep": self.minimum_keep,
        }

    def definition_hash(self) -> str:
        """Hash structural semantics without hashing learned checkpoint values."""

        payload = {
            "names": self.names,
            "mode": self.mode,
            "suppression_budget_per_gate": self.suppression_budget_per_gate,
            "minimum_keep": self.minimum_keep,
            "temperature": self.temperature,
            "seed": self.seed if self.mode == "frozen_random" else None,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()
