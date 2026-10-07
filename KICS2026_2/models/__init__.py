"""Neural-network modules used by PPO and the SRPO-inspired experiment."""

from .shared_actor_critic import SharedActorCritic
from .trajectory_encoder import TrajectoryEncoder, TrajectoryWorldModel

__all__ = ["SharedActorCritic", "TrajectoryEncoder", "TrajectoryWorldModel"]
