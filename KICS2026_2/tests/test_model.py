import torch

from models.shared_actor_critic import SharedActorCritic


def test_joint_log_probability_is_sum_of_agent_log_probabilities():
    torch.manual_seed(0)
    model = SharedActorCritic(player_observation_dim=5, hidden_sizes=(16, 8))
    observations = torch.randn(7, 2, 5)
    actions, joint_log_prob, entropy, values, logits = model.get_action_and_value(
        observations
    )
    expected = torch.log_softmax(logits, dim=-1).gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1).sum(dim=-1)
    assert actions.shape == (7, 2)
    assert logits.shape == (7, 2, 6)
    assert joint_log_prob.shape == entropy.shape == values.shape == (7,)
    assert torch.allclose(joint_log_prob, expected, atol=1e-6)


def test_evaluating_saved_actions_reproduces_log_probability():
    model = SharedActorCritic(player_observation_dim=4, hidden_sizes=(8,))
    observations = torch.randn(3, 2, 4)
    actions, original_log_prob, _, _, _ = model.get_action_and_value(observations)
    evaluated_log_prob, _, _, _ = model.evaluate_actions(observations, actions)
    assert torch.allclose(original_log_prob, evaluated_log_prob)
