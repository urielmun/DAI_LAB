import numpy as np
import torch

from collect_trajectories import (
    collect_episode,
    collect_trajectory_group,
    pad_trajectory_group,
)
from models.shared_actor_critic import SharedActorCritic


class FakeTeamEnv:
    def __init__(self, episode_length=3):
        self.episode_length = episode_length
        self.timestep = 0

    def _observation(self):
        return np.full((2, 4), self.timestep, dtype=np.float32)

    def reset(self, seed=None):
        self.timestep = 0
        return self._observation(), {"seed": seed}

    def step(self, action):
        assert np.asarray(action).shape == (2,)
        self.timestep += 1
        truncated = self.timestep >= self.episode_length
        info = {
            "sparse_reward": float(self.timestep == self.episode_length),
            "shaped_reward": 0.1,
            "delivery_count": int(truncated),
            "success": bool(truncated),
        }
        return self._observation(), info["sparse_reward"], False, truncated, info


def test_collector_stores_rollout_log_probs_and_padding():
    torch.manual_seed(4)
    env = FakeTeamEnv(episode_length=3)
    model = SharedActorCritic(4, hidden_sizes=(8,))
    trajectory = collect_episode(env, model, torch.device("cpu"), episode_seed=7)
    observations = torch.as_tensor(trajectory.observations[:-1])
    actions = torch.as_tensor(trajectory.actions)
    with torch.no_grad():
        recalculated, _, _, _ = model.evaluate_actions(observations, actions)
    assert np.allclose(trajectory.old_log_probs, recalculated.numpy(), atol=1e-6)
    assert trajectory.length == 3
    assert trajectory.success

    group = collect_trajectory_group(
        env, model, group_size=2, device=torch.device("cpu"), base_seed=10
    )
    batch = pad_trajectory_group(group, torch.device("cpu"))
    assert batch.observations.shape == (2, 3, 2, 4)
    assert batch.encoder_observations.shape == (2, 4, 2, 4)
    assert batch.valid_mask.all()
