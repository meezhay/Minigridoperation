import numpy as np
import pytest

from minigrid_mdp.agents.operator_rulebook import OperatorRulebook
from minigrid_mdp.env.minigrid_env import (
    MiniGridEnv,
    ACTION_CHARGE_FROM_DIESEL,
    ACTION_DISCHARGE_BATTERY,
    ACTION_SOLAR_ONLY,
)

NIGHT, SUNNY = 0.0, 20.0


def obs(soc, pv, diesel_on):
    return np.array([soc, pv, 5.0, diesel_on, 0.0, 1.0, 1.0], dtype=np.float32)


@pytest.mark.parametrize("soc, pv, diesel_on, expected", [
    (0.25, NIGHT, 0, ACTION_CHARGE_FROM_DIESEL),  # below 30%: start
    (0.40, NIGHT, 1, ACTION_CHARGE_FROM_DIESEL),  # running and below 50%: keep charging
    (0.40, NIGHT, 0, ACTION_DISCHARGE_BATTERY),   # off and above 30%: stay on battery
    (0.55, NIGHT, 1, ACTION_DISCHARGE_BATTERY),   # reached 50%: stop
    (0.60, SUNNY, 0, ACTION_SOLAR_ONLY),          # sun up: solar
])
def test_start_at_30_stop_at_50(soc, pv, diesel_on, expected):
    agent = OperatorRulebook(MiniGridEnv().action_space)
    assert agent.act(obs(soc, pv, diesel_on)) == expected