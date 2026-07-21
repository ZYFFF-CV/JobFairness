import numpy as np
import pandas as pd

from fuxictr_ext.fairjob.input_proxy_probe import (
    encode_inputs,
    fit_input_probe,
    select_feature_set,
)


def test_input_feature_controls_preserve_expected_columns():
    features = ["user_id", "product_id", "cat0", "senior", "num16"]
    assert select_feature_set(features, "ids_only") == ["user_id", "product_id"]
    assert select_feature_set(features, "non_id") == ["cat0", "senior", "num16"]
    assert "user_id" not in select_feature_set(features, "no_user_id")
    assert "product_id" not in select_feature_set(features, "no_product_id")


def test_train_only_target_encoding_handles_unseen_test_category():
    frames = {
        "train": pd.DataFrame({"cat0": ["a", "a", "b", "b"] * 20, "num16": range(80)}),
        "valid": pd.DataFrame({"cat0": ["a", "b"] * 10, "num16": range(20)}),
        "test": pd.DataFrame({"cat0": ["unseen", "a", "b", "unseen"] * 5, "num16": range(20)}),
    }
    targets = {
        "train": np.array([0, 0, 1, 1] * 20),
        "valid": np.array([0, 1] * 10),
        "test": np.array([0, 0, 1, 1] * 5),
    }
    encoded, metadata = encode_inputs(
        frames, targets, categorical=["cat0"], numeric=["num16"], seed=7
    )
    assert metadata["target_encoder_fit_split"] == "train"
    assert encoded["test"].shape == (20, 2)
    assert np.isfinite(encoded["test"]).all()
    result = fit_input_probe(encoded, targets, seed=7)
    assert 0.5 <= result["test_auc"] <= 1.0
