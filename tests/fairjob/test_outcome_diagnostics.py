import math

import pandas as pd

from fuxictr_ext.fairjob.outcome_diagnostics import (
    bootstrap_statistic,
    proxy_measurement_sensitivity,
)


def test_outcome_bootstrap_statistic_preserves_signed_dp():
    frame = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_pred": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
            "protected_attribute": [0, 0, 1, 1, 0, 1],
            "senior": [1, 1, 1, 1, 0, 0],
        }
    )
    result = bootstrap_statistic(frame)
    assert math.isclose(result["DP_signed"], 0.0, abs_tol=1e-12)
    assert result["DP_abs"] == abs(result["DP_signed"])
    assert 0.0 <= result["AUC"] <= 1.0


def test_proxy_measurement_sensitivity_is_deterministic():
    frame = pd.DataFrame(
        {
            "y_pred": [0.1, 0.8, 0.2, 0.7] * 20,
            "protected_attribute": [0, 0, 1, 1] * 20,
            "senior": [1, 1, 1, 1] * 20,
        }
    )
    first = proxy_measurement_sensitivity(frame, rates=(0.1,), repeats=3, seed=4)
    second = proxy_measurement_sensitivity(frame, rates=(0.1,), repeats=3, seed=4)
    assert first == second
    assert set(first["modes"]) == {
        "symmetric_flip",
        "group_0_to_1",
        "group_1_to_0",
        "random_missing",
    }
