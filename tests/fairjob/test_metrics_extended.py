import math

import pandas as pd

from fuxictr_ext.fairjob.calibration import brier_score, expected_calibration_error
from fuxictr_ext.fairjob.metrics_extended import (
    compute_extended_metrics,
    demographic_parity_details,
)


def make_frame():
    return pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1, 1, 0],
            "y_pred": [0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.6, 0.3],
            "protected_attribute": [0, 0, 1, 1, 0, 0, 1, 1],
            "senior": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )


def test_demographic_parity_details_preserve_sign():
    details = demographic_parity_details(make_frame())
    assert math.isclose(details["DP_group_0_mean"], 0.45)
    assert math.isclose(details["DP_group_1_mean"], 0.45)
    assert math.isclose(details["DP_signed"], 0.0, abs_tol=1e-12)
    assert details["DP_abs"] == abs(details["DP_signed"])
    assert details["DP_senior_n"] == 4


def test_calibration_metrics_are_hand_checkable():
    assert math.isclose(brier_score([0, 1], [0.25, 0.75]), 0.0625)
    assert math.isclose(expected_calibration_error([0, 1], [0.25, 0.75], 2), 0.25)


def test_extended_metrics_include_group_gaps():
    metrics = compute_extended_metrics(make_frame(), ece_bins=4)
    assert metrics["overall_n"] == 8
    assert metrics["group_0_n"] == 4
    assert metrics["group_1_n"] == 4
    assert "group_gap_ECE" in metrics
    assert metrics["overall_AUC"] is not None
