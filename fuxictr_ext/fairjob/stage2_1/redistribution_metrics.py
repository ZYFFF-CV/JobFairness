"""Descriptive Stage2.1 metrics for representation-graph redistribution."""

from __future__ import annotations

import math


def classify_delta(delta: float, material_delta: float) -> str:
    """Classify one paired decodability delta using the frozen threshold."""

    if material_delta <= 0:
        raise ValueError("material_delta must be positive.")
    value = float(delta)
    if not math.isfinite(value):
        raise ValueError("Redistribution deltas must be finite.")
    if value <= -material_delta:
        return "reduced"
    if value >= material_delta:
        return "amplified"
    return "stable"


def summarize_redistribution(
    deltas: dict[str, float],
    material_delta: float,
    epsilon: float = 1e-12,
) -> dict:
    """Summarize graph-wide reduction and migration without information claims."""

    if not deltas:
        raise ValueError("At least one representation delta is required.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    states = {
        name: classify_delta(value, material_delta)
        for name, value in deltas.items()
    }
    reduction_mass = sum(max(-float(value), 0.0) for value in deltas.values())
    migration_mass = sum(max(float(value), 0.0) for value in deltas.values())
    return {
        "material_delta": float(material_delta),
        "states": states,
        "reduction_count": sum(value == "reduced" for value in states.values()),
        "migration_count": sum(
            value == "amplified" for value in states.values()
        ),
        "stable_count": sum(value == "stable" for value in states.values()),
        "reduction_mass": float(reduction_mass),
        "migration_mass": float(migration_mass),
        "redistribution_index": float(
            migration_mass / (migration_mass + reduction_mass + epsilon)
        ),
    }


def protocol_consistency(
    row_deltas: dict[str, float],
    user_deltas: dict[str, float],
    material_delta: float,
) -> dict:
    """Compare row- and raw-user-disjoint directions target by target."""

    if set(row_deltas) != set(user_deltas):
        raise ValueError("Protocol deltas must contain identical targets.")
    row_summary = summarize_redistribution(row_deltas, material_delta)
    user_summary = summarize_redistribution(user_deltas, material_delta)
    agreement = {
        name: row_summary["states"][name] == user_summary["states"][name]
        for name in row_deltas
    }
    return {
        "row_disjoint": row_summary,
        "user_disjoint": user_summary,
        "state_agreement": agreement,
        "agreement_count": sum(agreement.values()),
        "target_count": len(agreement),
        "agreement_fraction": float(sum(agreement.values()) / len(agreement)),
    }
