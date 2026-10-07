import torch

from models.trajectory_encoder import TrajectoryEncoder, TrajectoryWorldModel
from rewards.self_reference_reward import compute_self_reference_scores


def test_encoder_ignores_padding_after_valid_length():
    torch.manual_seed(2)
    encoder = TrajectoryEncoder(8, hidden_dim=12, embedding_dim=5)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, 3:] = torch.randn_like(second[:, 3:]) * 100
    lengths = torch.tensor([3])
    assert torch.allclose(
        encoder(first, lengths), encoder(second, lengths), atol=1e-6
    )


def test_world_model_loss_is_finite():
    encoder = TrajectoryEncoder(8, hidden_dim=12, embedding_dim=5)
    world_model = TrajectoryWorldModel(encoder)
    observations = torch.randn(2, 4, 2, 4)
    next_observations = torch.randn(2, 4, 2, 4)
    actions = torch.randint(0, 6, (2, 4, 2))
    loss = world_model.self_supervised_loss(
        observations,
        next_observations,
        actions,
        lengths=torch.tensor([4, 2]),
        valid_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_nearer_failure_receives_higher_self_reference_score():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0]], dtype=torch.float32
    )
    result = compute_self_reference_scores(
        embeddings,
        success_mask=torch.tensor([True, False, False]),
        dbscan_min_samples=2,
    )
    assert result.valid
    assert result.scores[0] == 1.0
    assert result.scores[1] > result.scores[2]
    assert torch.all((result.scores >= 0) & (result.scores <= 1))


def test_no_success_group_is_explicitly_invalid():
    result = compute_self_reference_scores(
        torch.randn(4, 3), torch.zeros(4, dtype=torch.bool)
    )
    assert not result.valid
    assert result.skip_reason == "no_successful_trajectory"


def test_dbscan_success_center_path_is_used_when_cluster_exists():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]], dtype=torch.float32
    )
    result = compute_self_reference_scores(
        embeddings,
        success_mask=torch.tensor([True, True, False]),
        dbscan_eps=0.1,
        dbscan_min_samples=2,
    )
    assert result.valid
    assert result.center_method == "dbscan"
    assert result.center_count == 1
