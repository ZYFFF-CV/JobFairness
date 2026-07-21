import numpy as np

from fuxictr_ext.fairjob.proxy_probe import fit_probe


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
    assert result[0]["valid_auc"] > 0.95
    assert result[0]["test_auc"] > 0.95
