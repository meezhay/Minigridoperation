"""Original (v1) reward function, kept only to reproduce the load-shedding reward hack."""

from minigrid_mdp.env.minigrid_env import MiniGridEnv

V1_DIESEL_RUNTIME_PENALTY = 0.10  # was diesel_runtime_penalty in the v1 config


class V1RewardEnv(MiniGridEnv):
    """Corrected grid physics, original v1 reward. Shedding load costs nothing here."""

    def _compute_reward(self, flows: dict, load_kw: float) -> tuple[float, dict]:
        r   = self.rew_cfg
        die = self.die_cfg
        bat = self.bat_cfg

        unmet_kwh = flows["unmet_load_kw"] * self.dt
        r_unmet   = -r["unmet_load_cost_per_kwh"] * unmet_kwh

        diesel_kwh   = flows["diesel_kw"] * self.dt
        fuel_cost    = diesel_kwh * die["fuel_consumption_l_per_kwh"] * die["fuel_price_usd_per_litre"]
        startup_cost = die["startup_cost_usd"] if flows["diesel_on"] else 0.0
        fixed_cost   = die["hourly_fixed_cost_usd"] * self.dt if flows["diesel_on"] else 0.0
        r_fuel       = -(fuel_cost + startup_cost + fixed_cost) * r["fuel_cost_weight"]

        excess_dod   = max(0.0, (1.0 - self._soc) - bat["stress_threshold_dod"])
        r_bat_stress = -r["battery_stress_cost_per_dod"] * excess_dod

        pv_surplus    = max(0.0, flows["pv_kw"] - load_kw)
        r_diesel_idle = -V1_DIESEL_RUNTIME_PENALTY if flows["diesel_on"] and pv_surplus > 0 else 0.0

        total = r_unmet + r_fuel + r_bat_stress + r_diesel_idle
        return total, {
            "unmet_load":     r_unmet,
            "fuel":           r_fuel,
            "battery_stress": r_bat_stress,
            "diesel_idle":    r_diesel_idle,
        }