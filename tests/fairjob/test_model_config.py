from configs.fairjob.make_model_config import generate_entries


def test_stage1_1_model_matrix_is_complete():
    entries = generate_entries()
    assert len(entries) == 58
    assert "DeepFM_fairjob_pre_ranking_proxy_excluded_smoke" in entries
    assert "DCNv2_fairjob_post_display_proxy_included_full" in entries
    assert "DeepFM_fairjob_pre_ranking_no_user_id_proxy_excluded_full" in entries
    assert "DCNv2_fairjob_pre_ranking_no_identity_ids_proxy_excluded_full" in entries
    for expid, config in entries.items():
        if not expid.startswith(("M5DCNv2_", "Stage2DCNv2_")):
            assert config["dataset_id"] == expid.split("_", 1)[1]
        assert config["model"] in {
            "FairJobDeepFM",
            "FairJobDCNv2",
            "FairJobMitigatedDCNv2",
            "FairJobGraphContainmentDCNv2",
        }
        if config["model"] != "FairJobDeepFM":
            assert config["use_low_rank_mixture"] is False


def test_m5_configs_freeze_six_dcnv2_methods_for_both_modes():
    entries = generate_entries()
    m5 = {key: value for key, value in entries.items() if key.startswith("M5DCNv2_")}
    assert len(m5) == 12
    assert {config["mitigation_method"] for config in m5.values()} == {
        "baseline",
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    }
    for config in m5.values():
        assert config["model"] == "FairJobMitigatedDCNv2"
        assert config["dataset_id"].startswith(
            "fairjob_m5_pre_ranking_no_user_id_proxy_excluded_"
        )


def test_stage2_configs_freeze_graph_ablation_matrix():
    entries = generate_entries()
    stage2 = {
        key: value
        for key, value in entries.items()
        if key.startswith("Stage2DCNv2_")
    }
    assert len(stage2) == 18
    assert {config["stage2_method"] for config in stage2.values()} == {
        "baseline",
        "final_only_no_gate",
        "multi_layer_no_gate",
        "final_only_path_gate",
        "multi_layer_path_gate_full",
        "full_without_joint_adversary",
        "layer_risk_sum",
        "structured_random_gate",
        "matched_capacity_regularization",
    }
    for expid, config in stage2.items():
        assert config["model"] == "FairJobGraphContainmentDCNv2"
        assert config["dataset_id"].startswith(
            "fairjob_m5_pre_ranking_no_user_id_proxy_excluded_"
        )
        expected_teacher = (
            expid.endswith("_full")
            and config["stage2_method"] != "baseline"
        )
        assert config["stage2_require_baseline_teacher"] is expected_teacher
