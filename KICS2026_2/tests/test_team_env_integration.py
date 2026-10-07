import pytest


@pytest.mark.integration
def test_real_overcooked_wrapper_shapes_and_step():
    pytest.importorskip("overcooked_ai_py")
    from team_env import OvercookedTeamEnv

    env = OvercookedTeamEnv(horizon=3, reward_mode="shaped")
    observation, _ = env.reset(seed=1)
    assert observation.shape[0] == 2
    assert observation.dtype.name == "float32"
    for _ in range(3):
        observation, _reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
    assert terminated or truncated
    assert "delivery_count" in info
    env.close()
