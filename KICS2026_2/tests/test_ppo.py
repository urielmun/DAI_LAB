import torch

from algorithms.ppo import PPOBatch, PPOConfig, compute_gae, ppo_update
from models.shared_actor_critic import SharedActorCritic


def parameters_changed(before, module):
    return any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def test_compute_gae_matches_hand_calculation():
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 1.0]),
        values=torch.tensor([0.5, 0.25]),
        final_value=torch.tensor(0.0),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages, torch.tensor([1.5, 0.75]))
    assert torch.allclose(returns, torch.tensor([2.0, 1.0]))


def test_ppo_update_changes_actor_and_critic():
    torch.manual_seed(1)
    model = SharedActorCritic(4, hidden_sizes=(16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randn(32, 2, 4)
    with torch.no_grad():
        actions, old_log_probs, _, values, _ = model.get_action_and_value(observations)
    advantages = torch.linspace(-1.0, 1.0, 32)
    batch = PPOBatch(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values,
        advantages=advantages,
        returns=values + advantages,
    )
    actor_before = [p.detach().clone() for p in model.actor.parameters()]
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    stats = ppo_update(
        model,
        optimizer,
        batch,
        PPOConfig(update_epochs=2, minibatch_size=16, target_kl=None),
    )
    assert parameters_changed(actor_before, model.actor)
    assert parameters_changed(critic_before, model.critic)
    assert torch.isfinite(torch.tensor(stats.total_loss))
