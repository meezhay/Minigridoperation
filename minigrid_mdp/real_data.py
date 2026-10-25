"""Load real hourly solar and demand series for the data-driven environment."""
import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760


def load_pvgis_per_kwp(path, utc_offset_hours=3):
    """PVGIS hourly output (W from 1 kWp, UTC) → kW per kWp, shifted to local time. Losses are already included."""
    watts = pd.read_csv(path).iloc[:, 1].to_numpy(dtype=float)
    _check_length(watts, path)
    return np.roll(watts / 1000.0, utc_offset_hours)


def load_hourly_load_kw(path):
    """Hourly demand in kW, assumed to start at local midnight on 1 January."""
    kw = pd.read_csv(path).iloc[:, 1].to_numpy(dtype=float)
    _check_length(kw, path)
    return kw


def _check_length(values, path):
    if len(values) != HOURS_PER_YEAR:
        raise ValueError(f"{path}: expected {HOURS_PER_YEAR} hourly values, got {len(values)}")