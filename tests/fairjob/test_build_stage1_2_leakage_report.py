from fuxictr_ext.fairjob.build_stage1_2_leakage_report import _classify


def _item(mean, negative=0, positive=0):
    return {
        "mean": mean,
        "negative_seeds": negative,
        "positive_seeds": positive,
    }


def _deltas(final_linear=-0.01, final_nonlinear=-0.02):
    payload = {}
    for representation in (
        "embedding_flat",
        "cross_layer_0",
        "cross_layer_1",
        "cross_layer_2",
        "dcnv2_final_pre_mitigation",
    ):
        for family in ("linear", "nonlinear"):
            payload[(representation, family)] = _item(0.0)
    payload[("dcnv2_final", "linear")] = _item(
        final_linear, negative=3 if final_linear < 0 else 0
    )
    payload[("dcnv2_final", "nonlinear")] = _item(
        final_nonlinear, negative=3 if final_nonlinear < 0 else 0
    )
    return payload


def test_classify_clean_reduction_and_redistribution():
    clean = _deltas()
    assert _classify(clean)[0] == "leakage_reduced"

    migrated = _deltas()
    migrated[("cross_layer_2", "nonlinear")] = _item(0.01, positive=3)
    classification, evidence = _classify(migrated)
    assert classification == "redistributed"
    assert evidence[0]["representation"] == "cross_layer_2"


def test_classify_probe_family_disagreement_and_no_reduction():
    disagreement = _deltas(final_linear=-0.003, final_nonlinear=0.004)
    assert _classify(disagreement)[0] == "inconclusive_probe_family_disagreement"

    unchanged = _deltas(final_linear=-0.001, final_nonlinear=-0.002)
    assert _classify(unchanged)[0] == "not_reduced"
