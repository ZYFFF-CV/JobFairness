import pytest
import torch

from fuxictr_ext.fairjob.structured_path_gates import StructuredPathGates


NAMES = ["cross_0", "cross_1", "dnn", "fusion"]


def test_uniform_learned_gate_starts_with_exact_total_budget():
    gates = StructuredPathGates(
        NAMES,
        mode="learned",
        suppression_budget_per_gate=0.1,
        minimum_keep=0.5,
    )
    telemetry = gates.telemetry()
    assert telemetry["actual_total_suppression"] == pytest.approx(0.4)
    assert set(telemetry["keep"]) == set(NAMES)
    assert all(
        value == pytest.approx(0.9)
        for value in telemetry["keep"].values()
    )


def test_gate_is_scalar_dimension_preserving_and_trainable():
    gates = StructuredPathGates(NAMES, mode="learned")
    value = torch.ones(3, 5, requires_grad=True)
    output = gates.apply("dnn", value)
    assert output.shape == value.shape
    output.sum().backward()
    assert value.grad is not None
    assert gates.allocation_logits.grad is not None


def test_identity_gate_preserves_values_exactly():
    gates = StructuredPathGates(NAMES, mode="identity")
    value = torch.randn(4, 7)
    assert torch.equal(gates.apply("fusion", value), value)


def test_frozen_random_gate_is_seed_deterministic_and_budget_matched():
    first = StructuredPathGates(NAMES, mode="frozen_random", seed=23)
    second = StructuredPathGates(NAMES, mode="frozen_random", seed=23)
    third = StructuredPathGates(NAMES, mode="frozen_random", seed=24)
    first_keep = torch.stack(list(first.keep_values().values()))
    second_keep = torch.stack(list(second.keep_values().values()))
    third_keep = torch.stack(list(third.keep_values().values()))
    assert torch.equal(first_keep, second_keep)
    assert not torch.equal(first_keep, third_keep)
    assert first.telemetry()["nominal_total_suppression"] == pytest.approx(
        third.telemetry()["nominal_total_suppression"]
    )
    assert first.telemetry()["actual_total_suppression"] == pytest.approx(
        first.telemetry()["nominal_total_suppression"]
    )
    assert len(first.definition_hash()) == 64


def test_concentrated_allocation_retains_budget_and_minimum_keep():
    gates = StructuredPathGates(
        NAMES,
        mode="learned",
        suppression_budget_per_gate=0.1,
        minimum_keep=0.8,
    )
    with torch.no_grad():
        gates.allocation_logits.copy_(torch.tensor([20.0, -20.0, -20.0, -20.0]))
    telemetry = gates.telemetry()
    assert telemetry["actual_total_suppression"] == pytest.approx(0.4)
    assert min(telemetry["keep"].values()) >= 0.8 - 1e-7
