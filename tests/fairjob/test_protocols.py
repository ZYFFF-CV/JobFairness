from pathlib import Path

import yaml

from fuxictr_ext.fairjob.protocols import features_for_protocol, load_protocols


ROOT = Path(__file__).resolve().parents[2]


def load_manifest():
    return yaml.safe_load(
        (ROOT / "configs/fairjob/feature_schema.yaml").read_text(encoding="utf-8")
    )


def test_pre_ranking_excludes_post_display_fields():
    task, fairness = load_protocols(
        ROOT / "configs/fairjob/task_protocols.yaml",
        ROOT / "configs/fairjob/fairness_protocols.yaml",
    )
    features, protocol_name, regime_name = features_for_protocol(
        load_manifest(), task, fairness, "pre_ranking", "proxy_excluded"
    )
    assert protocol_name == "pre-ranking"
    assert regime_name == "proxy-excluded"
    assert "rank" not in features
    assert "displayrandom" not in features
    assert "impression_id" not in features
    assert "protected_attribute_feat" not in features
    assert {"user_id", "product_id", "senior", "cat0", "num16"}.issubset(features)


def test_proxy_included_changes_only_protected_proxy():
    task, fairness = load_protocols(
        ROOT / "configs/fairjob/task_protocols.yaml",
        ROOT / "configs/fairjob/fairness_protocols.yaml",
    )
    manifest = load_manifest()
    excluded, _, _ = features_for_protocol(
        manifest, task, fairness, "post_display", "proxy_excluded"
    )
    included, _, _ = features_for_protocol(
        manifest, task, fairness, "post_display", "proxy_included"
    )
    assert set(included).difference(excluded) == {"protected_attribute_feat"}
    assert set(excluded).difference(included) == set()


def test_identity_control_protocols_remove_only_requested_ids():
    task, fairness = load_protocols(
        ROOT / "configs/fairjob/task_protocols.yaml",
        ROOT / "configs/fairjob/fairness_protocols.yaml",
    )
    manifest = load_manifest()
    base, _, _ = features_for_protocol(
        manifest, task, fairness, "pre_ranking", "proxy_excluded"
    )
    no_user, _, _ = features_for_protocol(
        manifest, task, fairness, "pre_ranking_no_user_id", "proxy_excluded"
    )
    no_product, _, _ = features_for_protocol(
        manifest, task, fairness, "pre_ranking_no_product_id", "proxy_excluded"
    )
    no_ids, _, _ = features_for_protocol(
        manifest, task, fairness, "pre_ranking_no_identity_ids", "proxy_excluded"
    )
    assert set(base).difference(no_user) == {"user_id"}
    assert set(base).difference(no_product) == {"product_id"}
    assert set(base).difference(no_ids) == {"user_id", "product_id"}
