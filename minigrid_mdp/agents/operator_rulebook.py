"""Rulebook based on real EMONE site practice: start diesel below 30% SOC, charge to 50%, then stop."""
from minigrid_mdp.agents.rule_based_agent import RuleBasedAgent
from minigrid_mdp.env.minigrid_env import (
    ACTION_CHARGE_FROM_DIESEL,
    ACTION_DISCHARGE_BATTERY,
    ACTION_SOLAR_ONLY,
)


class OperatorRulebook(RuleBasedAgent):
    agent_type = "operator_rulebook"

    START_SOC = 0.30
    STOP_SOC = 0.50
    IDX_DIESEL = 3

    def act(self, obs):
        soc = float(obs[self.IDX_SOC])
        pv = float(obs[self.IDX_PV])
        diesel_running = obs[self.IDX_DIESEL] > 0.5

        if soc < self.START_SOC or (diesel_running and soc < self.STOP_SOC):
            return ACTION_CHARGE_FROM_DIESEL
        if pv >= self.pv_negligible_kw:
            return ACTION_SOLAR_ONLY
        return ACTION_DISCHARGE_BATTERY