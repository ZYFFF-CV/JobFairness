import numpy as np
import pytest

from fuxictr_ext.fairjob.stage2_1.probe_audit import (
    conditional_gain,
    fit_probe_suite,
    joint_gain,
)


def _protocol():
    return {
        "probe_families": {
            "linear": {"C": [0.1], "max_iter": 200},
            "matched_capacity_nonlinear": {
                "hidden_layer_sizes": [[4]],
                "alpha": [0.0001],
                "max_iter": 30,
                "early_stopping": True,
            },
            "independent_bounded": {
                "n_estimators": 10,
                "max_depth": [3],
                "min_samples_leaf": 2,
                "n_jobs": 1,
            },
        }
    }


def _fixed():
    return {
        "linear": {"C": 0.1},
        "matched_capacity_nonlinear": {
            "hidden_layer_sizes": [4],
            "alpha": 0.0001,
        },
        "independent_bounded": {"max_depth": 3},
    }


def _data(seed=7):
    rng = np.random.default_rng(seed)
    arrays = []
    for rows in (160, 80, 80):
        X = rng.normal(size=(rows, 3)).astype(np.float32)
        y = (X[:, 0] + 0.2 * rng.normal(size=rows) > 0).astype(np.int8)
        arrays.extend([X, y])
    return arrays


def test_probe_suite_reports_selected_fixed_and_cross_entropy_endpoints():
    X_train, y_train, X_valid, y_valid, X_test, y_test = _data()
    result = fit_probe_suite(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        _protocol(),
        fixed_capacity=_fixed(),
        seed=11,
    )
    assert set(result["families"]) == {
        "linear",
        "matched_capacity_nonlinear",
        "independent_bounded",
    }
    for family in result["families"].values():
        assert family["auc_selected"]["test_auc"] >= 0.5
        assert family["ce_selected"]["test_cross_entropy"] > 0
        assert family["fixed_capacity"]["params"]
        assert len(family["candidate_validation"]) == 1


def test_conditional_and_joint_gain_use_held_out_cross_entropy():
    def suite(value):
        return {
            "families": {
                family: {
                    role: {"test_cross_entropy": value}
                    for role in ("ce_selected", "fixed_capacity")
                }
                for family in (
                    "linear",
                    "matched_capacity_nonlinear",
                    "independent_bounded",
                )
            }
        }

    conditional = conditional_gain(suite(0.6), suite(0.5))
    assert (
        conditional["linear"]["ce_selected"]["conditional_predictive_gain"]
        == pytest.approx(0.1)
    )
    joint = joint_gain(suite(0.6), suite(0.55), suite(0.5))
    assert joint["linear"]["fixed_capacity"][
        "joint_predictive_gain"
    ] == pytest.approx(0.05)
