import torch
import pytest

from fuxictr_ext.fairjob.adaptive_constraints import (
    AdaptiveDualController,
    GraphRiskAggregator,
    proxy_advantages,
)


def test_smooth_max_proxy_advantage_emphasizes_lowest_bce():
    easy = torch.tensor(0.2, requires_grad=True)
    hard = torch.tensor(0.7, requires_grad=True)
    aggregator = GraphRiskAggregator(
        mode="smooth_max_proxy_advantage", temperature=0.05
    )
    objective, telemetry = aggregator({"easy": easy, "hard": hard})
    objective.backward()
    assert easy.grad > hard.grad
    assert telemetry["worst_node"] == "easy"
    assert proxy_advantages({"easy": easy})["easy"] > 0


def test_equal_losses_keep_normalized_soft_min_on_original_scale():
    losses = {
        "left": torch.tensor(0.4),
        "right": torch.tensor(0.4),
    }
    objective, _ = GraphRiskAggregator(
        mode="smooth_max_proxy_advantage", temperature=0.1
    )(losses)
    assert torch.allclose(objective, torch.tensor(0.4))


def test_cvar_uses_lowest_bce_tail():
    losses = {
        "a": torch.tensor(0.1),
        "b": torch.tensor(0.2),
        "c": torch.tensor(0.8),
        "d": torch.tensor(0.9),
    }
    objective, telemetry = GraphRiskAggregator(
        mode="cvar_low_bce", cvar_fraction=0.5
    )(losses)
    assert torch.allclose(objective, torch.tensor(0.15))
    assert telemetry["worst_node"] == "a"


def test_adaptive_dual_projects_updates_to_nonnegative_range():
    controller = AdaptiveDualController(
        ["node", "joint"],
        {"node": 0.1, "joint": 0.2},
        learning_rate=0.5,
        maximum=1.0,
    )
    first = controller.update({"node": 0.3, "joint": 0.1})
    assert first["node"] == pytest.approx(0.1)
    assert first["joint"] == 0.0
    second = controller.update({"node": 3.0, "joint": 3.0})
    assert second == {"node": 1.0, "joint": 1.0}
