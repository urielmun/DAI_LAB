"""Neural observation n-gram encoder with a trained penultimate representation.

context_size=2 means [o[t-1], o[t]] -> o[t+1] (a neural trigram).
The predictor is observation-only; unlike the original GRU world model, it
does not condition on actions. MSE therefore predicts the conditional mean
over the data policy's possible next actions. This is an explicit baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NGramTrajectoryEncoder(nn.Module):
    """Pool trained local-window hidden states into one trajectory vector."""

    def __init__(
        self,
        joint_observation_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        context_size: int = 2,
    ) -> None:
        super().__init__()
        if min(joint_observation_dim, hidden_dim, embedding_dim, context_size) <= 0:
            raise ValueError("All encoder dimensions and context_size must be positive")
        self.joint_observation_dim = int(joint_observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.context_size = int(context_size)
        self.feature_net = nn.Sequential(
            nn.Linear(self.context_size * self.joint_observation_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.GELU(),
        )

    def encode_sequence(
        self, observations: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return [B,T,E] local hidden states, with invalid positions zeroed.

        Windows are causal, ordered from oldest to newest, and never cross
        episode boundaries. Missing left context repeats that episode's o[0].
        """
        if observations.ndim == 4:
            observations = observations.flatten(start_dim=2)
        if observations.ndim != 3 or observations.shape[-1] != self.joint_observation_dim:
            raise ValueError("Expected observations [B,T,2,D] or [B,T,joint_dim]")
        batch, time, _ = observations.shape
        lengths = lengths.to(observations.device)
        if lengths.shape != (batch,) or torch.any(lengths < 1) or torch.any(lengths > time):
            raise ValueError("lengths must have shape [B] and values in [1,T]")
        if lengths.is_floating_point() or lengths.dtype == torch.bool:
            raise ValueError("lengths must be integer counts")
        valid = torch.arange(time, device=observations.device)[None, :] < lengths[:, None]
        observations = observations.float().masked_fill(~valid[..., None], 0.0)
        offsets = torch.arange(1 - self.context_size, 1, device=observations.device)
        indices = (torch.arange(time, device=observations.device)[:, None] + offsets).clamp_min(0)
        windows = observations[:, indices, :].flatten(start_dim=2)
        hidden = self.feature_net(windows)
        return hidden.masked_fill(~valid[..., None], 0.0)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean-pool all valid local features, then L2-normalize [B,E].

        Local order is retained inside a window. Global episode order is not
        retained by mean pooling. No untrained projection is appended.
        """
        hidden = self.encode_sequence(observations, lengths)
        mean_hidden = hidden.sum(dim=1) / lengths.to(hidden.device)[:, None]
        return F.normalize(mean_hidden, p=2, dim=-1)

    def freeze(self) -> NGramTrajectoryEncoder:
        self.eval()
        self.requires_grad_(False)
        return self


class NGramWorldModel(nn.Module):
    """Next-observation regression through exactly the reward encoder features."""

    def __init__(self, encoder: NGramTrajectoryEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        # A single linear output layer: its input is exactly the local hidden
        # state that is pooled for the SRPO trajectory representation.
        self.next_observation_head = nn.Linear(
            encoder.embedding_dim, encoder.joint_observation_dim
        )

    def predict_next_observations(
        self, observations: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder.encode_sequence(observations, lengths)
        return self.next_observation_head(hidden)

    def self_supervised_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del actions  # Accepted for the shared trainer API; observation-only model.
        predictions = self.predict_next_observations(observations, lengths)
        targets = next_observations.flatten(start_dim=2) if next_observations.ndim == 4 else next_observations
        if targets.shape != predictions.shape or valid_mask.shape != predictions.shape[:2]:
            raise ValueError("Target or mask shape does not match predictions")
        expected_mask = torch.arange(predictions.shape[1], device=predictions.device)[None, :] < lengths.to(predictions.device)[:, None]
        if not torch.equal(valid_mask.bool().to(predictions.device), expected_mask):
            raise ValueError("valid_mask and lengths disagree")
        # Select before squaring, so arbitrary padding (even NaN) is excluded.
        return F.mse_loss(predictions[expected_mask], targets.float()[expected_mask])
