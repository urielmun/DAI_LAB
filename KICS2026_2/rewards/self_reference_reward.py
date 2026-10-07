"""Trajectory-level self-reference scores for the SRPO-inspired experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelfReferenceResult:
    """Scores plus diagnostics needed for transparent experiment logging."""

    scores: torch.Tensor
    nearest_squared_distances: torch.Tensor
    success_count: int
    failure_count: int
    center_count: int
    center_method: str
    valid: bool
    skip_reason: str | None
    failed_distance_mean: float
    failed_distance_std: float


def build_success_centers(
    success_embeddings: torch.Tensor,
    dbscan_eps: float = 0.35,
    dbscan_min_samples: int = 2,
) -> tuple[torch.Tensor, str]:
    """Cluster successful embeddings and return normalized cluster means.

    Small groups often produce no DBSCAN cluster.  In that case every success
    embedding becomes a center.  This fallback is an explicit Overcooked
    adaptation, not an exact reproduction of the original SRPO experiment.
    """

    if success_embeddings.ndim != 2 or success_embeddings.shape[0] == 0:
        raise ValueError("success_embeddings must be non-empty [N,E]")
    if dbscan_eps <= 0 or dbscan_min_samples <= 0:
        raise ValueError("DBSCAN parameters must be positive")

    if success_embeddings.shape[0] < dbscan_min_samples:
        return F.normalize(success_embeddings, dim=-1), "success_fallback"

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for DBSCAN. Install the environment.yml."
        ) from error

    labels_np = DBSCAN(
        eps=float(dbscan_eps),
        min_samples=int(dbscan_min_samples),
        metric="euclidean",
    ).fit_predict(success_embeddings.detach().cpu().numpy())
    cluster_labels = sorted(int(label) for label in np.unique(labels_np) if label >= 0)
    if not cluster_labels:
        return F.normalize(success_embeddings, dim=-1), "success_fallback"

    labels = torch.as_tensor(labels_np, device=success_embeddings.device)
    centers = torch.stack(
        [success_embeddings[labels == label].mean(dim=0) for label in cluster_labels]
    )
    return F.normalize(centers, dim=-1), "dbscan"


def compute_self_reference_scores(
    embeddings: torch.Tensor,
    success_mask: torch.Tensor,
    failure_score_scale: float = 0.8,
    dbscan_eps: float = 0.35,
    dbscan_min_samples: int = 2,
    eps: float = 1e-8,
) -> SelfReferenceResult:
    """Assign success score 1 and distance-based scores to failures.

    For failed trajectory ``i``:

    ``d_i = min_c ||h_i - c||^2``

    ``g_i = alpha * sigmoid(-(d_i - mean(d)) / (std(d) + eps))``

    The minus sign is essential: a smaller distance must produce a larger score.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [group, embedding_dim]")
    if success_mask.ndim != 1 or success_mask.shape[0] != embeddings.shape[0]:
        raise ValueError("success_mask must have shape [group]")
    if not 0.0 < failure_score_scale < 1.0:
        raise ValueError("failure_score_scale must be between 0 and 1")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf")

    success_mask = success_mask.bool()
    failure_mask = ~success_mask
    success_count = int(success_mask.sum().item())
    failure_count = int(failure_mask.sum().item())
    scores = torch.zeros(
        embeddings.shape[0], device=embeddings.device, dtype=embeddings.dtype
    )
    distances = torch.full_like(scores, float("nan"))

    if success_count == 0:
        return SelfReferenceResult(
            scores=scores,
            nearest_squared_distances=distances,
            success_count=0,
            failure_count=failure_count,
            center_count=0,
            center_method="none",
            valid=False,
            skip_reason="no_successful_trajectory",
            failed_distance_mean=float("nan"),
            failed_distance_std=float("nan"),
        )
    scores[success_mask] = 1.0
    if failure_count == 0:
        return SelfReferenceResult(
            scores=scores,
            nearest_squared_distances=distances,
            success_count=success_count,
            failure_count=0,
            center_count=success_count,
            center_method="not_needed",
            valid=False,
            skip_reason="no_failed_trajectory",
            failed_distance_mean=float("nan"),
            failed_distance_std=float("nan"),
        )

    success_embeddings = F.normalize(embeddings[success_mask], dim=-1)
    failed_embeddings = F.normalize(embeddings[failure_mask], dim=-1)
    centers, center_method = build_success_centers(
        success_embeddings,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    squared_distances = torch.cdist(failed_embeddings, centers, p=2).pow(2)
    nearest_distances = squared_distances.min(dim=1).values
    distance_mean = nearest_distances.mean()
    distance_std = nearest_distances.std(unbiased=False)
    if float(distance_std.item()) <= eps:
        normalized_distances = torch.zeros_like(nearest_distances)
    else:
        normalized_distances = (
            nearest_distances - distance_mean
        ) / (distance_std + eps)
    failed_scores = failure_score_scale * torch.sigmoid(-normalized_distances)
    scores[failure_mask] = failed_scores
    distances[failure_mask] = nearest_distances

    return SelfReferenceResult(
        scores=scores,
        nearest_squared_distances=distances,
        success_count=success_count,
        failure_count=failure_count,
        center_count=int(centers.shape[0]),
        center_method=center_method,
        valid=True,
        skip_reason=None,
        failed_distance_mean=float(distance_mean.item()),
        failed_distance_std=float(distance_std.item()),
    )
