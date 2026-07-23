import numpy as np
import pandas as pd

from fuxictr_ext.fairjob.outcome_floor_audit import (
    calibration_curve,
    classify_dp_floor,
    context_conditioned_dp,
)


def test_context_conditioned_dp_reconstructs_standardized_means():
    result = context_conditioned_dp(
        {
            "contexts": [
                {
                    "reference_weight": 0.25,
                    "group_0_mean": 0.1,
                    "group_1_mean": 0.2,
                    "group_0_n": 10,
                    "group_1_n": 12,
                },
                {
                    "reference_weight": 0.75,
                    "group_0_mean": 0.3,
                    "group_1_mean": 0.2,
                    "group_0_n": 20,
                    "group_1_n": 18,
                },
            ],
            "common_contexts": 2,
            "composition_residual_signed": 0.01,
        }
    )
    assert np.isclose(result["DP_group_0_mean"], 0.25)
    assert np.isclose(result["DP_group_1_mean"], 0.2)
    assert np.isclose(result["DP_signed"], -0.05)
    assert result["DP_group_0_n"] == 30
    assert result["DP_group_1_n"] == 30


def test_calibration_curve_preserves_rows_and_bin_means():
    frame = pd.DataFrame(
        {"y_true": [0, 1, 1, 0], "y_pred": [0.05, 0.15, 0.85, 1.0]}
    )
    curve = calibration_curve(frame, bins=2)
    assert sum(row["n"] for row in curve) == len(frame)
    assert np.isclose(curve[0]["prediction_mean"], 0.1)
    assert np.isclose(curve[1]["positive_rate"], 0.5)


def test_dp_floor_classification_uses_seed_noise_and_bootstrap_zero_crossing():
    runs = []
    for seed, value, interval in (
        (2019, 0.00004, (-0.0002, 0.0002)),
        (2020, 0.00035, (0.0001, 0.0008)),
        (2021, 0.00113, (0.0007, 0.0014)),
    ):
        protocols = {
            protocol: {"DP_signed": -value, "DP_abs": value}
            for protocol in (
                "all_logged",
                "random_display",
                "context_conditioned",
            )
        }
        runs.append(
            {
                "method": "baseline",
                "seed": seed,
                "protocols": protocols,
                "cluster_bootstrap": {
                    "all_logged": {
                        "metrics": {
                            "DP_signed": {
                                "ci_low": interval[0],
                                "ci_high": interval[1],
                            }
                        }
                    }
                },
            }
        )
    result = classify_dp_floor(runs)
    assert result["classification"] == "DP_near_floor_for_current_setting"
    assert result["baseline_bootstrap_intervals_crossing_zero"] == 1
