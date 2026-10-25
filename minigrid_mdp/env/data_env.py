"""Mini-grid environment driven by real hourly solar and demand series instead of formulas."""
import numpy as np

from minigrid_mdp.env.minigrid_env import MiniGridEnv
from minigrid_mdp.real_data import HOURS_PER_YEAR


class DataDrivenEnv(MiniGridEnv):
    """Same grid, actions and reward; solar and demand come from a year of real hourly data."""

    def __init__(self, pv_per_kwp, load_kw, *args, days_per_episode=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._pv_series = pv_per_kwp * self.pv_cfg["rated_power_kw"]
        self._load_series = load_kw
        self._days_per_episode = days_per_episode
        self.ep_len = 24 * days_per_episode
        self._start_hour = 0

    def reset(self, *, seed=None, options=None):
        self._start_hour = 24 * self._pick_start_day(seed, options)
        return super().reset(seed=seed, options=options)

    def _pick_start_day(self, seed, options):
        if options and "start_day" in options:
            return int(options["start_day"])
        rng = np.random.default_rng(seed) if seed is not None else self._np_rng
        return int(rng.integers(365 - self._days_per_episode + 1))

    def _current_index(self):
        return (self._start_hour + self._step_count) % HOURS_PER_YEAR

    def _pv_output(self, hour):
        return float(self._pv_series[self._current_index()])

    def _load_demand(self, hour):
        return float(self._load_series[self._current_index()])