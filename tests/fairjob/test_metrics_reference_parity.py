import math

import pandas as pd

from fuxictr_ext.fairjob.metrics import (
    FEMALE_RATIO,
    MALE_RATIO,
    compute_fairjob_metrics,
    demographic_parity,
    utility,
    utility_product,
)


def make_frames():
    """Build a synthetic FairJob-like batch with hand-checkable metric values.

    The fixture includes two protected groups in the senior scope for DP, three
    impressions for U, and three products for U_TILDE's product averaging path.
    """

    meta = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4, 5],
            "click": [0, 1, 1, 0, 1, 0],
            "protected_attribute": [0, 1, 0, 1, 0, 1],
            "senior": [1, 1, 1, 1, 0, 0],
            "displayrandom": [1, 1, 1, 1, 1, 1],
            "impression_id": [10, 10, 11, 11, 11, 12],
            "product_id": [100, 100, 101, 101, 101, 102],
        }
    )
    pred = pd.DataFrame(
        {
            "row_id": meta["row_id"],
            "model": "SYN",
            "regime": "unaware",
            "mode": "smoke",
            "y_true": meta["click"],
            "y_pred": [0.1, 0.9, 0.2, 0.3, 0.1, 0.7],
        }
    )
    frame = meta.copy()
    frame["y_pred"] = pred["y_pred"]
    return pred, meta, frame


def test_demographic_parity():
    _, _, frame = make_frames()
    assert math.isclose(demographic_parity(frame), 0.45)


def test_utility():
    _, _, frame = make_frames()
    # Impression 10: [0, 2] mean=1. Impression 11: [2, 0, 1] mean=1. Impression 12: 0.
    assert math.isclose(utility(frame), 2.0 / 3.0)


def test_utility_product_with_ratio():
    _, _, frame = make_frames()
    # Product 100 has the clicked protected group 1 row at rank 2; product 101
    # has the clicked protected group 0 row at rank 3. Product 102 contributes 0.
    expected_product_100 = 2.0 * MALE_RATIO
    expected_product_101 = 3.0 * FEMALE_RATIO
    expected = (expected_product_100 + expected_product_101 + 0.0) / 3.0
    assert math.isclose(utility_product(frame, unbiased_ratio=True), expected)


def test_compute_metrics_keys_are_finite():
    pred, meta, _ = make_frames()
    metrics = compute_fairjob_metrics(pred, meta)
    for key in ["NLLH", "AUC", "AVG_P_SCORE", "DP", "U", "U_TILDE"]:
        assert math.isfinite(metrics[key])
