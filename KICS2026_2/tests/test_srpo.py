import copy

import torch

from algorithms.srpo import (
    SRPOBatch,
    SRPOConfig,
    compute_exact_joint_reference_kl,
    compute_group_advantages,
    srpo_update,
)
from models.shared_actor_critic import SharedActorCritic


def clone_parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def any_changed(before, module):
    return any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def test_group_advantages_have_zero_mean_and_unit_population_std():
    advantages = compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6
    )


def test_reference_kl_is_zero_for_identical_logits():
    logits = torch.randn(5, 2, 6)
    kl = compute_exact_joint_reference_kl(logits, logits.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_srpo_updates_actor_but_not_critic_or_reference():
    torch.manual_seed(3)
    model = SharedActorCritic(4, hidden_sizes=(16,))
    reference = copy.deepcopy(model)
    reference.eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    observations = torch.randn(3, 4, 2, 4)
    flat_observations = observations.reshape(-1, 2, 4)
    with torch.no_grad():
        actions, old_log_probs, _, _, _ = model.get_action_and_value(flat_observations)
    batch = SRPOBatch(
        observations=observations,
        actions=actions.reshape(3, 4, 2),
        old_log_probs=old_log_probs.reshape(3, 4),
        valid_mask=torch.ones(3, 4, dtype=torch.bool),
    )
    actor_before = clone_parameters(model.actor)
    critic_before = clone_parameters(model.critic)
    reference_before = clone_parameters(reference)
    stats = srpo_update(
        model,
        reference,
        optimizer,
        batch,
        compute_group_advantages(torch.tensor([0.1, 0.5, 1.0])),
        SRPOConfig(update_epochs=2, minibatch_size=6, target_old_kl=None),
    )
    assert stats.update_applied
    assert any_changed(actor_before, model.actor)
    assert not any_changed(critic_before, model.critic)
    assert not any_changed(reference_before, reference)
