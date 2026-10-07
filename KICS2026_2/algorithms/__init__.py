"""Policy-update algorithms.  PPO and SRPO are intentionally separate."""

from .ppo import PPOBatch, PPOConfig, PPOStats, compute_gae, ppo_update
from .srpo import (
    SRPOBatch,
    SRPOConfig,
    SRPOStats,
    compute_group_advantages,
    srpo_update,
)

__all__ = [
    "PPOBatch",
    "PPOConfig",
    "PPOStats",
    "SRPOBatch",
    "SRPOConfig",
    "SRPOStats",
    "compute_gae",
    "compute_group_advantages",
    "ppo_update",
    "srpo_update",
]
