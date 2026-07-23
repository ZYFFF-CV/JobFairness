import numpy as np

from fuxictr_ext.fairjob.prediction_collapse_diagnostics import (
    _nllh,
    score_distribution,
    select_calibrator,
)


def test_validation_only_calibration_returns_finite_probabilities():
    rng = np.random.default_rng(7)
    probability = np.linspace(0.01, 0.99, 200)
    y_true = rng.binomial(1, probability)
    result = select_calibrator(probability, y_true, seed=11)
    calibrated = result.pop("model")(probability)
    assert result["selected"] in {"identity", "temperature", "platt", "beta", "isotonic"}
    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert result["fit_rows"] == result["selection_rows"] == 100


def test_distribution_reports_prediction_collapse_signals():
    y_true = np.array([0, 0, 1, 1])
    informative = score_distribution(y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    collapsed = score_distribution(y_true, np.array([0.49, 0.5, 0.5, 0.51]))
    assert collapsed["prediction_std"] < informative["prediction_std"]
    assert collapsed["class_score_gap"] < informative["class_score_gap"]
    assert _nllh(y_true, np.array([0.1, 0.2, 0.8, 0.9])) < _nllh(
        y_true, np.full(4, 0.5)
    )
