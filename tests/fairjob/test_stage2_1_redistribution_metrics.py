import math

import pytest

from fuxictr_ext.fairjob.stage2_1.redistribution_metrics import (
    classify_delta,
    protocol_consistency,
    summarize_redistribution,
)


def test_delta_classification_uses_frozen_inclusive_material_threshold():
    assert classify_delta(-0.005, 0.005) == "reduced"
    assert classify_delta(0.004, 0.005) == "stable"
    assert classify_delta(0.005, 0.005) == "amplified"
    with pytest.raises(ValueError):
        classify_delta(float("nan"), 0.005)


def test_redistribution_mass_and_index_are_descriptive():
    result = summarize_redistribution(
        {"reduced": -0.02, "stable": 0.001, "amplified": 0.03},
        material_delta=0.005,
    )
    assert result["reduction_count"] == 1
    assert result["migration_count"] == 1
    assert result["stable_count"] == 1
    assert math.isclose(result["reduction_mass"], 0.02)
    assert math.isclose(result["migration_mass"], 0.031)
    assert 0.6 < result["redistribution_index"] < 0.7


def test_protocol_consistency_requires_identical_targets():
    result = protocol_consistency(
        {"a": -0.01, "b": 0.01},
        {"a": -0.02, "b": 0.001},
        material_delta=0.005,
    )
    assert result["agreement_count"] == 1
    assert result["agreement_fraction"] == 0.5
    with pytest.raises(ValueError):
        protocol_consistency({"a": 0.1}, {"b": 0.1}, 0.005)
