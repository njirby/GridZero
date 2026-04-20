"""Shape and correctness tests for the GSPO loss function."""
import pytest


def test_gspo_loss_is_scalar():
    torch = pytest.importorskip("torch")
    from gridzero.training.gspo import gspo_loss

    G, B = 8, 4
    log_probs = torch.randn(B * G)
    old_log_probs = torch.randn(B * G)
    rewards = torch.randn(B * G)
    group_ids = torch.repeat_interleave(torch.arange(B), G)

    loss, metrics = gspo_loss(log_probs, old_log_probs, rewards, group_ids)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert "loss" in metrics
    assert "clip_fraction" in metrics


def test_gspo_loss_no_nan_with_equal_probs():
    torch = pytest.importorskip("torch")
    from gridzero.training.gspo import gspo_loss

    probs = torch.zeros(16)
    rewards = torch.ones(16)
    group_ids = torch.repeat_interleave(torch.arange(2), 8)
    loss, _ = gspo_loss(probs, probs, rewards, group_ids)
    assert not torch.isnan(loss)


def test_group_normalize():
    torch = pytest.importorskip("torch")
    from gridzero.training.gspo import _group_normalize

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1])
    advantages = _group_normalize(rewards, group_ids)
    # Group 0 should have mean ~0
    assert advantages[:4].mean().abs() < 1e-5
    # Group 1: only 2 elements, std=5, advantages should be ±1
    assert advantages[4:].abs().allclose(torch.ones(2))
