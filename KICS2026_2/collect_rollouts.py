"""Deprecated compatibility alias.

The canonical implementation is ``collect_trajectories.py``.  New code must
import ``collect_trajectory_group`` from there.  This file contains no duplicate
logic and exists only because an earlier project tree used this old name.
"""

from collect_trajectories import (
    Trajectory,
    TrajectoryGroupBatch,
    append_group_summary_jsonl,
    collect_episode,
    collect_trajectory_group,
    pad_trajectory_group,
    save_trajectory_npz,
)

__all__ = [
    "Trajectory",
    "TrajectoryGroupBatch",
    "append_group_summary_jsonl",
    "collect_episode",
    "collect_trajectory_group",
    "pad_trajectory_group",
    "save_trajectory_npz",
]
