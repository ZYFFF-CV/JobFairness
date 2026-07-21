from configs.fairjob.make_model_config import generate_entries


def test_stage1_1_model_matrix_is_complete():
    entries = generate_entries()
    assert len(entries) == 16
    assert "DeepFM_fairjob_pre_ranking_proxy_excluded_smoke" in entries
    assert "DCNv2_fairjob_post_display_proxy_included_full" in entries
    for expid, config in entries.items():
        assert config["dataset_id"] == expid.split("_", 1)[1]
        assert config["model"] in {"FairJobDeepFM", "FairJobDCNv2"}
        assert config["use_low_rank_mixture"] is False if expid.startswith("DCNv2") else True
