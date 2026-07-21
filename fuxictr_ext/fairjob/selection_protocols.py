"""Selection scopes used by FairJob Stage1.1 evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def select_rows(
    frame: pd.DataFrame,
    protocol: str,
    context: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return rows eligible under a declared logging protocol.

    ``position_corrected`` cannot be represented by a row filter alone and is
    therefore handled by :func:`position_weights`.
    """

    if protocol == "all_logged":
        selected = frame
    elif protocol == "random_display":
        selected = frame[frame["displayrandom"].astype(int) == 1]
    elif protocol == "context_conditioned":
        if not context:
            raise ValueError("context_conditioned requires at least one filter.")
        mask = np.ones(len(frame), dtype=bool)
        for column, value in context.items():
            if column not in frame.columns:
                raise KeyError(f"Unknown context column: {column}")
            mask &= frame[column].to_numpy() == value
        selected = frame[mask]
    elif protocol == "position_corrected":
        selected = frame
    else:
        raise KeyError(f"Unknown selection protocol: {protocol}")
    if selected.empty:
        raise ValueError(f"Selection protocol {protocol} produced no rows.")
    return selected.copy()


def position_weights(
    frame: pd.DataFrame,
    propensity_column: str = "logging_propensity",
    min_propensity: float = 1e-3,
) -> np.ndarray:
    """Return normalized inverse-propensity weights for position correction.

    Stage1.1 does not infer propensities from outcomes. Callers must supply an
    externally estimated logging propensity column, otherwise the protocol
    fails explicitly instead of silently claiming a corrected estimate.
    """

    if propensity_column not in frame.columns:
        raise ValueError(
            f"position_corrected requires column {propensity_column!r}."
        )
    propensity = frame[propensity_column].to_numpy(dtype=float)
    if not np.isfinite(propensity).all() or (propensity <= 0).any():
        raise ValueError("Logging propensities must be finite and positive.")
    weights = 1.0 / np.maximum(propensity, min_propensity)
    return weights / weights.mean()
