import numpy as np

from fuxictr_ext.fairjob.proxy_probe import add_amplification, fit_probe


def test_probe_selects_on_validation_and_reports_test():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(200, 4))
    X_valid = rng.normal(size=(80, 4))
    X_test = rng.normal(size=(80, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    y_valid = (X_valid[:, 0] > 0).astype(int)
    y_test = (X_test[:, 0] > 0).astype(int)
    result = fit_probe(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        probe_type="linear",
        seed=7,
    )
    assert len(result) == 1
    assert result[0]["family"] == "linear"
    assert result[0]["converged"]
    assert result[0]["n_iter"] > 0
    assert result[0]["valid_auc"] > 0.95
    assert result[0]["test_auc"] > 0.95


def test_amplification_matches_stage1_1_definition():
    probe = {"test_auc": 0.8}
    add_amplification(probe, input_baseline_auc=0.7)
    assert np.isclose(probe["leakage_amplification"], 0.1)
    assert np.isclose(probe["input_normalized_amplification"], 1.5)
