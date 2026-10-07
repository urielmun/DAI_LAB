"""Gymnasium adapter for a two-agent Overcooked team.

The policy observation has shape ``[2, player_observation_dim]``.  The two
rows are the lossless encodings from each player's perspective.  A shared
actor consumes one row at a time; a centralized critic may flatten both rows.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

OVERCOOKED_SOURCE_COMMIT = "739950a079cdaed5a44fcc662efc40244c205d06"


class OvercookedTeamEnv(gym.Env[np.ndarray, np.ndarray]):
    """Control both chefs with one joint action.

    Args:
        layout_name: Name of an Overcooked-AI layout.
        horizon: Maximum number of environment steps per episode.
        target_deliveries: Deliveries required for ``success`` and binary reward.
        reward_mode: ``sparse``, ``shaped``, or terminal ``binary`` reward.
        shaping_coef: Multiplier for the environment's shaped reward component.

    Notes:
        The upstream environment combines MDP termination and horizon expiry in
        one ``done`` value.  This adapter reports horizon expiry as Gymnasium
        ``truncated=True``.  The training code deliberately treats the fixed
        horizon as the end of the finite-horizon task and therefore does not
        bootstrap through it by default.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        layout_name: str = "cramped_room",
        horizon: int = 400,
        target_deliveries: int = 1,
        reward_mode: str = "sparse",
        shaping_coef: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if target_deliveries <= 0:
            raise ValueError("target_deliveries must be positive")
        if reward_mode not in {"sparse", "shaped", "binary"}:
            raise ValueError("reward_mode must be sparse, shaped, or binary")

        self.layout_name = layout_name
        self.horizon = int(horizon)
        self.target_deliveries = int(target_deliveries)
        self.reward_mode = reward_mode
        self.shaping_coef = float(shaping_coef)

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp,
            horizon=self.horizon,
            info_level=0,
        )
        self.action_space = spaces.MultiDiscrete(
            np.asarray([Action.NUM_ACTIONS, Action.NUM_ACTIONS], dtype=np.int64)
        )

        self.base_env.reset(regen_mdp=False)
        example_observation = self.get_player_observations()
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=example_observation.shape,
            dtype=np.float32,
        )
        self._delivery_count = 0
        self._episode_finished = False

    @property
    def player_observation_dim(self) -> int:
        """Flattened lossless observation size for one player."""

        return int(np.prod(self.observation_space.shape[1:]))

    @property
    def joint_observation_dim(self) -> int:
        """Flattened size of both player-relative observations."""

        return int(np.prod(self.observation_space.shape))

    def get_player_observations(self) -> np.ndarray:
        """Return player-relative lossless encodings as ``float32 [2, D]``."""

        encoded = self.base_env.lossless_state_encoding_mdp(self.base_env.state)
        return np.stack(
            [np.asarray(player_view, dtype=np.float32).reshape(-1) for player_view in encoded],
            axis=0,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.base_env.reset(regen_mdp=False)
        self._delivery_count = 0
        self._episode_finished = False
        observation = self.get_player_observations()
        info = {
            "layout_name": self.layout_name,
            "horizon": self.horizon,
            "target_deliveries": self.target_deliveries,
            "delivery_count": 0,
            "success": False,
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_finished:
            raise RuntimeError("Episode finished. Call reset() before step().")

        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError(f"Invalid joint action: {action_array!r}")

        joint_action = tuple(
            Action.INDEX_TO_ACTION[int(action_index)]
            for action_index in action_array
        )
        _, sparse_team_reward, upstream_done, upstream_info = self.base_env.step(
            joint_action
        )

        previous_delivery_count = self._delivery_count
        self._delivery_count = int(
            sum(
                len(player_events)
                for player_events in self.base_env.game_stats["soup_delivery"]
            )
        )
        step_delivery_count = self._delivery_count - previous_delivery_count
        success = self._delivery_count >= self.target_deliveries

        shaped_by_agent = np.asarray(
            upstream_info["shaped_r_by_agent"], dtype=np.float32
        )
        sparse_by_agent = np.asarray(
            upstream_info["sparse_r_by_agent"], dtype=np.float32
        )
        shaped_team_reward = float(shaped_by_agent.sum())

        mdp_terminal = bool(self.mdp.is_terminal(self.base_env.state))
        horizon_reached = bool(self.base_env.state.timestep >= self.horizon)
        terminated = mdp_terminal
        truncated = bool(upstream_done and not mdp_terminal) or horizon_reached
        self._episode_finished = terminated or truncated

        if self.reward_mode == "binary":
            training_reward = float(self._episode_finished and success)
        elif self.reward_mode == "shaped":
            training_reward = (
                float(sparse_team_reward)
                + self.shaping_coef * shaped_team_reward
            )
        else:
            training_reward = float(sparse_team_reward)

        info: dict[str, Any] = {
            "sparse_reward": float(sparse_team_reward),
            "shaped_reward": shaped_team_reward,
            "sparse_reward_by_agent": sparse_by_agent,
            "shaped_reward_by_agent": shaped_by_agent,
            "delivery_count": self._delivery_count,
            "step_delivery_count": step_delivery_count,
            "success": success,
            "timestep": int(self.base_env.state.timestep),
            "upstream_done": bool(upstream_done),
            "end_reason": (
                "mdp_terminal"
                if terminated
                else "horizon"
                if truncated
                else None
            ),
        }
        return (
            self.get_player_observations(),
            float(training_reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """The upstream simulator owns no external process to close."""

        return


if __name__ == "__main__":
    environment = OvercookedTeamEnv(horizon=10, reward_mode="shaped")
    observation, reset_info = environment.reset(seed=42)
    print("observation shape:", observation.shape)
    print("reset info:", reset_info)
    for _ in range(10):
        observation, reward, terminated, truncated, info = environment.step(
            environment.action_space.sample()
        )
        if terminated or truncated:
            print("final info:", info)
            break
