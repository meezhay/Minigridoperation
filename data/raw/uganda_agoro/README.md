# Agoro mini-grid data (northern Uganda)

**Source:** SEL-Columbia, [mini-grid_generation_system_design](https://github.com/SEL-Columbia/mini-grid_generation_system_design), the code and data for Wu et al., "Leveraging load shedding and daytime loads to improve mini-grid economics", *Energy Policy*.

**Used with the permission of the authors** (email from Yuezi Wu, October 2026).

| File | Contents |
| --- | --- |
| `lamwo_solar_2019.csv` | Hourly solar energy for Lamwo district, 2019, in Wh per 1,000 W of installed PV (PVGIS). Timestamps are UTC; `real_data.py` shifts them +3 hours to local time (EAT). |
| `agoro_fixed_loads.csv` | Hourly village demand, households and productive use. Assumed to be kW, in local time, starting at midnight on 1 January (awaiting confirmation). |
| `agoro_curtailable_loads.csv` | Hourly curtailable (flexible) demand. Not yet used. |