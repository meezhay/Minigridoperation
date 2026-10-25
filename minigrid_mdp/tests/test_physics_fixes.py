import pytest

from minigrid_mdp.env.minigrid_env import (
    MiniGridEnv,
    ACTION_CHARGE_FROM_DIESEL,
    ACTION_RUN_DIESEL_ONLY,
)


def make_env_at_min_soc():
    env = MiniGridEnv()
    env.reset(seed=0)
    env._soc = env.bat_cfg["min_soc"]
    return env


def test_pv_peak_reaches_rated_times_pr():
    env = MiniGridEnv()
    env.reset(seed=0)
    expected_peak = env.pv_cfg["rated_power_kw"] * env.pv_cfg["performance_ratio"]
    samples = [env._pv_output(13) for _ in range(200)]
    assert max(samples) == pytest.approx(expected_peak)
    assert max(samples) > 30.0


@pytest.mark.parametrize("action", [ACTION_RUN_DIESEL_ONLY, ACTION_CHARGE_FROM_DIESEL])
def test_empty_battery_shortfall_is_unmet_not_free(action):
    env = make_env_at_min_soc()
    diesel_rated = env.die_cfg["rated_power_kw"]
    load_kw = diesel_rated + 10.0
    flows = env._dispatch(action, pv_kw=0.0, load_kw=load_kw, hour=20)
    assert flows["battery_kw"] == pytest.approx(0.0)
    assert flows["unmet_load_kw"] == pytest.approx(10.0)


def test_startup_cost_charged_once_per_start():
    env = MiniGridEnv()
    env.reset(seed=0)
    first = env.step(ACTION_RUN_DIESEL_ONLY)[4]["flows"]
    second = env.step(ACTION_RUN_DIESEL_ONLY)[4]["flows"]
    assert first["diesel_started"] is True
    assert second["diesel_started"] is False