from collections import OrderedDict

import torch

from fuxictr.features import FeatureMap
from fuxictr_ext.fairjob.models import FairJobDCNv2
from fuxictr_ext.fairjob.representation_graph import (
    build_joint_representations,
    graph_hash,
    load_representation_graph,
    validate_graph_representations,
)


def _feature_map(tmp_path):
    feature_map = FeatureMap("stage2_graph_test", str(tmp_path))
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
            )
        ]
    )
    feature_map.labels = ["click"]
    feature_map.default_emb_dim = 4
    feature_map.num_fields = feature_map.get_num_fields()
    return feature_map


def _model(tmp_path):
    return FairJobDCNv2(
        _feature_map(tmp_path),
        model_id="stage2_graph_test",
        gpu=-1,
        model_structure="parallel",
        use_low_rank_mixture=False,
        parallel_dnn_hidden_units=[8, 4],
        stacked_dnn_hidden_units=[8, 4],
        num_cross_layers=3,
        embedding_dim=4,
        optimizer="adam",
        loss="binary_crossentropy",
        learning_rate=1e-3,
        task="binary_classification",
        monitor="AUC",
        monitor_mode="max",
        metrics=["AUC"],
        model_root=str(tmp_path),
        verbose=0,
    )


def _batch():
    return {
        "product_id": torch.tensor([1, 2, 3, 4]),
        "click": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


def test_parallel_dcnv2_exposes_real_graph_nodes_without_prediction_change(tmp_path):
    model = _model(tmp_path)
    native = model(_batch())["y_pred"]
    diagnostic = model.forward_with_representations(_batch())
    graph = load_representation_graph()
    audit = validate_graph_representations(
        diagnostic["representations"], graph
    )
    assert torch.allclose(native, diagnostic["y_pred"], atol=1e-7)
    assert audit["rows"] == 4
    assert len(audit["node_shapes"]) == len(graph["nodes"])
    assert len(graph_hash(graph)) == 64
    assert torch.equal(
        diagnostic["representations"]["cross_path_output"],
        diagnostic["representations"]["cross_layer_2"],
    )
    assert torch.equal(
        diagnostic["representations"]["fusion_pre_logit"],
        diagnostic["representations"]["dcnv2_final"],
    )


def test_joint_paths_use_frozen_member_order_and_dimensions(tmp_path):
    model = _model(tmp_path)
    representations = model.forward_with_representations(_batch())[
        "representations"
    ]
    graph = load_representation_graph()
    joints = build_joint_representations(representations, graph)
    expected = torch.cat(
        [
            representations["cross_path_output"],
            representations["parallel_dnn_path_output"],
        ],
        dim=-1,
    )
    assert torch.equal(joints["joint_cross_dnn_outputs"], expected)
    assert joints["joint_cross_block_increments"].shape[-1] == 12
