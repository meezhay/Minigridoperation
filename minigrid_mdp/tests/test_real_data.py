from pathlib import Path

import pytest

from minigrid_mdp.real_data import load_pvgis_per_kwp, load_hourly_load_kw

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "uganda_agoro"
needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="Agoro data not downloaded")


@needs_data
def test_solar_peaks_at_local_noon():
    pv = load_pvgis_per_kwp(DATA_DIR / "lamwo_solar_2019.csv")
    mean_by_hour = pv.reshape(365, 24).mean(axis=0)
    assert 11 <= mean_by_hour.argmax() <= 13
    assert pv.max() <= 1.0


@needs_data
def test_load_daily_energy_is_plausible():
    load = load_hourly_load_kw(DATA_DIR / "agoro_fixed_loads.csv")
    assert 50 < load.sum() / 365 < 200