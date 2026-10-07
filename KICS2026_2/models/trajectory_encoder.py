"""GRU-based trajectory representation model.

GPU is an execution device, not an encoder architecture.  The architecture in
this project is a GRU sequence encoder.  Moving it and its tensors to CUDA makes
that GRU run on a GPU.

The original VLA SRPO paper uses a frozen, pretrained world-model encoder.  This
small GRU is an Overcooked-specific replacement and must be pretrained and then
frozen before its distances are treated as meaningful research signals.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TrajectoryEncoder(nn.Module):
    """Compress a variable-length joint-observation sequence into one vector."""

    def __init__(
        self,
        joint_observation_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(joint_observation_dim, hidden_dim, embedding_dim) <= 0:
            raise ValueError("Encoder dimensions must be positive")
        self.joint_observation_dim = int(joint_observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(self.joint_observation_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def _flatten_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 4:
            observations = observations.flatten(start_dim=2)
        if observations.ndim != 3:
            raise ValueError(
                "observations must have shape [B,T,2,D] or [B,T,joint_dim]"
            )
        if observations.shape[-1] != self.joint_observation_dim:
            raise ValueError(
                f"Expected joint observation dim {self.joint_observation_dim}, "
                f"received {observations.shape[-1]}"
            )
        return observations.float()

    def encode_sequence(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded GRU outputs and the last valid hidden state."""

        observations = self._flatten_observations(observations)
        if lengths.ndim != 1 or lengths.shape[0] != observations.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        if torch.any(lengths <= 0) or torch.any(lengths > observations.shape[1]):
            raise ValueError("Every length must be in [1, sequence_length]")

        projected = self.input_projection(observations)
        packed = pack_padded_sequence(
            projected,
            lengths.detach().to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, final_hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=observations.shape[1],
        )
        return outputs, final_hidden[-1]

    def forward(
        self,
        observations: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return L2-normalized trajectory embeddings ``[batch, embedding_dim]``."""

        _, final_hidden = self.encode_sequence(observations, lengths)
        embeddings = self.embedding_head(final_hidden)
        return F.normalize(embeddings, p=2, dim=-1)

    def freeze(self) -> TrajectoryEncoder:
        """Put the encoder in evaluation mode and disable gradients."""

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class TrajectoryWorldModel(nn.Module):
    """Action-conditioned next-observation predictor for encoder pretraining.

    This is a compact self-supervised pretraining objective, not a replacement
    for the large pretrained video world model used by the original SRPO work.
    """

    def __init__(self, encoder: TrajectoryEncoder, action_dim: int = 6) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_dim = int(action_dim)
        self.num_agents = 2
        predictor_input_dim = (
            self.encoder.hidden_dim + self.num_agents * self.action_dim
        )
        self.next_observation_head = nn.Sequential(
            nn.Linear(predictor_input_dim, self.encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.encoder.hidden_dim,
                self.encoder.joint_observation_dim,
            ),
        )

    def predict_next_observations(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flattened ``o[t+1]`` for every padded sequence position."""

        if actions.ndim != 3 or actions.shape[-1] != self.num_agents:
            raise ValueError("actions must have shape [B,T,2]")
        hidden_sequence, _ = self.encoder.encode_sequence(observations, lengths)
        if tuple(actions.shape[:2]) != tuple(hidden_sequence.shape[:2]):
            raise ValueError("Observation and action sequence shapes do not match")
        one_hot_actions = F.one_hot(
            actions.long(), num_classes=self.action_dim
        ).float().flatten(start_dim=2)
        predictor_inputs = torch.cat([hidden_sequence, one_hot_actions], dim=-1)
        return self.next_observation_head(predictor_inputs)

    def self_supervised_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked MSE next-observation prediction loss."""

        predictions = self.predict_next_observations(
            observations, actions, lengths
        )
        if next_observations.ndim == 4:
            targets = next_observations.flatten(start_dim=2)
        elif next_observations.ndim == 3:
            targets = next_observations
        else:
            raise ValueError("next_observations has an invalid shape")
        targets = targets.float()
        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes do not match")
        if valid_mask.shape != predictions.shape[:2]:
            raise ValueError("valid_mask must have shape [B,T]")
        per_step_loss = (predictions - targets).pow(2).mean(dim=-1)
        mask = valid_mask.float()
        return (per_step_loss * mask).sum() / mask.sum().clamp_min(1.0)
