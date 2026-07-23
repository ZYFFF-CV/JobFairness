import torch

from fuxictr_ext.fairjob.multinode_adversary import (
    MultiNodeAdversary,
    balanced_binary_loss,
    gradient_reverse,
)


def test_balanced_binary_loss_averages_groups_equally():
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0, 1.0])
    result = balanced_binary_loss(logits, target)
    expected_group_0 = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:2], target[:2]
    )
    expected_group_1 = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[2:], target[2:]
    )
    assert torch.allclose(result, (expected_group_0 + expected_group_1) / 2)


def test_gradient_reverse_changes_only_backward_direction():
    inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    output = gradient_reverse(inputs, scale=0.5)
    assert torch.equal(output, inputs)
    output.sum().backward()
    assert torch.equal(inputs.grad, torch.full_like(inputs, -0.5))


def test_multinode_adversary_checks_dimensions_and_backpropagates():
    module = MultiNodeAdversary(
        {"cross": 3, "joint": 5},
        hidden_units=(4,),
        gradient_reversal_scale=1.0,
    )
    representations = {
        "cross": torch.randn(8, 3, requires_grad=True),
        "joint": torch.randn(8, 5, requires_grad=True),
    }
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    logits = module(representations)
    losses = module.losses(logits, target)
    assert set(losses) == {"cross", "joint"}
    sum(losses.values()).backward()
    assert representations["cross"].grad is not None
    assert representations["joint"].grad is not None
