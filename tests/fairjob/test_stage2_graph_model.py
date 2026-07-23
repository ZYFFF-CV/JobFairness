from collections import OrderedDict

import torch

from fuxictr.features import FeatureMap
from fuxictr_ext.fairjob.models import (
    FairJobDCNv2,
    FairJobGraphContainmentDCNv2,
)


def _feature_map(tmp_path):
    feature_map = FeatureMap("stage2_model_test", str(tmp_path))
    feature_map.features = OrderedDict(
        [
            (
                "product_id",
                {
                    "type": "categorical",
                    "source": "",
                    "vocab_size": 12,
                    "padding_idx": 0,
                },
            ),
            ("protected_attribute_meta", {"type": "meta"}),
        ]
    )
    feature_map.labels = ["click"]
    feature_map.default_emb_dim = 8
    feature_map.num_fields = feature_map.get_num_fields()
    return feature_map


def _params(tmp_path):
    return {
        "model_id": "stage2_test",
        "gpu": -1,
        "model_structure": "parallel",
        "use_low_rank_mixture": False,
        "parallel_dnn_hidden_units": [12, 8],
        "stacked_dnn_hidden_units": [12, 8],
        "num_cross_layers": 2,
        "embedding_dim": 8,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "learning_rate": 1e-3,
        "task": "binary_classification",
        "monitor": "AUC",
        "monitor_mode": "max",
        "metrics": ["AUC"],
        "model_root": str(tmp_path),
        "verbose": 0,
        "seed": 2019,
    }


def _batch(include_proxy=True):
    batch = {
        "product_id": torch.tensor([1, 2, 3, 4, 5]),
        "click": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    }
    if include_proxy:
        batch["protected_attribute_meta"] = torch.tensor([0, 1, 0, 1, 0])
    return batch


def test_stage2_baseline_is_prediction_identical_to_native_adapter(tmp_path):
    params = _params(tmp_path)
    stage2 = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path), stage2_method="baseline", **params
    )
    native = FairJobDCNv2(_feature_map(tmp_path), **params)
    native.load_state_dict(stage2.state_dict(), strict=False)
    stage2.eval()
    native.eval()
    assert torch.allclose(
        stage2(_batch(include_proxy=False))["y_pred"],
        native(_batch(include_proxy=False))["y_pred"],
        atol=1e-7,
    )


def test_full_method_backpropagates_through_gates_and_adversaries(tmp_path):
    model = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path),
        stage2_method="multi_layer_path_gate_full",
        stage2_adversary_nodes=[
            "cross_layer_0",
            "cross_layer_1",
            "parallel_dnn_path_output",
            "fusion_pre_logit",
        ],
        **_params(tmp_path),
    )
    model.train()
    batch = _batch()
    output = model(batch)
    assert "joint_cross_dnn_outputs" in output["proxy_logits"]
    loss = model.compute_loss(output, model.get_labels(batch))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.path_gates.allocation_logits.grad is not None
    assert model.fc.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.proxy_adversary.parameters()
    )
    telemetry = model._stage2_last_telemetry
    assert telemetry["graph_risk"]["worst_node"] in output["proxy_logits"]
    assert telemetry["class_score_gap"] is not None


def test_stage2_inference_has_no_proxy_dependency(tmp_path):
    model = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path),
        stage2_method="multi_layer_path_gate_full",
        **_params(tmp_path),
    )
    model.eval()
    output = model(_batch(include_proxy=False))
    assert output["y_pred"].shape == (5, 1)
    assert "proxy_logits" not in output


def test_matched_capacity_control_blocks_adversary_encoder_gradient(tmp_path):
    model = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path),
        stage2_method="matched_capacity_regularization",
        **_params(tmp_path),
    )
    assert model.proxy_adversary.gradient_reversal_scale == 0.0
    assert model.path_gates.mode == "identity"
    assert len(model.proxy_adversary.adversaries) > 1


def test_frozen_baseline_teacher_initializes_student_and_stays_in_eval(tmp_path):
    params = _params(tmp_path)
    native = FairJobDCNv2(_feature_map(tmp_path), **params)
    checkpoint = tmp_path / "baseline.model"
    torch.save(native.state_dict(), checkpoint)
    model = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path),
        stage2_method="multi_layer_path_gate_full",
        stage2_backbone_checkpoint=str(checkpoint),
        stage2_require_baseline_teacher=True,
        **params,
    )
    model.train()
    assert model.teacher_backbone.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.teacher_backbone.parameters()
    )
    output = model(_batch())
    assert "frozen_teacher_logit" in output
    loss = model.compute_loss(output, model.get_labels(_batch()))
    assert loss.isfinite()
    assert model._stage2_last_telemetry["teacher_logit_scale"] >= 0.1


def test_preservation_loss_is_stable_under_large_common_logit_scale(tmp_path):
    model = FairJobGraphContainmentDCNv2(
        _feature_map(tmp_path),
        stage2_method="baseline",
        **_params(tmp_path),
    )
    student = torch.tensor([[0.0], [1000.0], [-1000.0], [500.0]])
    teacher = student + torch.tensor([[0.0], [20.0], [-20.0], [10.0]])
    return_dict = {
        "stage2_representations": {"dcnv2_logit": student},
        "frozen_teacher_logit": teacher,
    }
    centered, ranking, scale = model._preservation_losses(return_dict)
    assert centered < 0.01
    assert ranking.isfinite()
    assert scale > 100.0
