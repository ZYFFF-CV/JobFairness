import json
from collections import OrderedDict
from pathlib import Path

import torch

from configs.fairjob.make_dataset_config import feature_cols
from fuxictr.features import FeatureMap
from fuxictr_ext.fairjob.models import FairJobMitigatedDCNv2


ROOT = Path(__file__).resolve().parents[2]


def test_m5_meta_aliases_are_raw_and_excluded_from_model_inputs():
    manifest = {
        "categorical_id_features": ["product_id"],
        "categorical_features": ["cat0"],
        "binary_context_features": ["senior"],
        "protected_feature_for_model": "protected_attribute_feat",
        "numeric_features": ["num16"],
    }
    columns = feature_cols(
        manifest,
        ["product_id", "cat0", "senior", "num16"],
        include_fairness_meta=True,
    )
    specs = {
        name: column
        for column in columns
        for name in (column["name"] if isinstance(column["name"], list) else [column["name"]])
    }
    assert "protected_attribute_feat" not in specs
    for name, source in (
        ("protected_attribute_meta", "protected_attribute"),
        ("senior_meta", "senior"),
    ):
        assert specs[name]["type"] == "meta"
        assert specs[name]["remap"] is False
        assert specs[name]["preprocess"] == f"copy_from({source})"


def test_selective_masks_match_frozen_m3_dimensions():
    payload = json.loads(
        (ROOT / "configs/fairjob/m5_selective_indices.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["representation"] == "dcnv2_final"
    assert payload["dimensions"] == 928
    assert payload["masked_dimensions"] == 93
    assert set(payload["indices_by_seed"]) == {"2019", "2020", "2021"}
    for indices in payload["indices_by_seed"].values():
        assert len(indices) == 93
        assert len(set(indices)) == 93
        assert min(indices) >= 0
        assert max(indices) < 928


def _feature_map(tmp_path):
    feature_map = FeatureMap("m5_test", str(tmp_path))
    feature_map.features = OrderedDict(
        [
            (
                "product_id",
                {
                    "type": "categorical",
                    "source": "",
                    "vocab_size": 8,
                    "padding_idx": 0,
                },
            ),
            ("protected_attribute_meta", {"type": "meta"}),
            ("senior_meta", {"type": "meta"}),
        ]
    )
    feature_map.labels = ["click"]
    feature_map.default_emb_dim = 16
    feature_map.num_fields = feature_map.get_num_fields()
    return feature_map


def _model(tmp_path, method, selective_path=None):
    params = {
        "model_id": f"test_{method}",
        "gpu": -1,
        "model_structure": "parallel",
        "use_low_rank_mixture": False,
        "parallel_dnn_hidden_units": [32, 16],
        "stacked_dnn_hidden_units": [32, 16],
        "num_cross_layers": 2,
        "embedding_dim": 16,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "learning_rate": 1e-3,
        "task": "binary_classification",
        "monitor": "AUC",
        "monitor_mode": "max",
        "metrics": ["AUC"],
        "model_root": str(tmp_path),
        "verbose": 0,
        "mitigation_method": method,
        "seed": 2019,
    }
    if selective_path is not None:
        params["selective_indices_config"] = str(selective_path)
    return FairJobMitigatedDCNv2(_feature_map(tmp_path), **params)


def _batch():
    return {
        "product_id": torch.tensor([1, 2, 3, 4]),
        "protected_attribute_meta": torch.tensor([0, 1, 0, 1]),
        "senior_meta": torch.tensor([1, 1, 1, 1]),
        "click": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


def test_suppression_controls_use_declared_mask_sizes(tmp_path):
    # One 16-d embedding plus a 16-d parallel DNN gives a 32-d final state.
    selective_path = tmp_path / "indices.json"
    selective_path.write_text(
        json.dumps({"dimensions": 32, "indices_by_seed": {"2019": [1, 3, 5]}}),
        encoding="utf-8",
    )
    selective = _model(tmp_path, "selective_suppression", selective_path)
    matched = _model(tmp_path, "matched_random_suppression")
    global_control = _model(tmp_path, "global_suppression")
    assert int((selective.suppression_mask == 0).sum()) == 3
    assert int((matched.suppression_mask == 0).sum()) == 3
    assert int((global_control.suppression_mask == 0).sum()) == 16


def test_fairness_baselines_backpropagate_without_meta_input_leakage(tmp_path):
    batch = _batch()
    for method in ("dp_regularization", "adversarial"):
        model = _model(tmp_path, method)
        assert set(model.get_inputs(batch)) == {"product_id"}
        output = model(batch)
        loss = model.compute_loss(output, model.get_labels(batch))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.fc.weight.grad is not None
