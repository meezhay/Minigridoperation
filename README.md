# Mini-Grid Dispatch RL

A Gymnasium simulation of a **solar–battery–diesel mini-grid**, first for a simulated Nigerian village and then on a year of real data from a village in Uganda, and an honest test of whether a reinforcement-learning agent (PPO) can operate it better than a fixed rulebook.


**Short answer:** not meaningfully. On a simulated village, a PPO agent trained on week-long episodes spends about 6% less on fuel than a simple rulebook. On a year of **real data from Agoro, Uganda**, with outages priced the way operators treat them, a trained agent **ties** with a one-line rule taken from real microgrid practice. The originally published figure (26% fewer diesel hours) came from a simulation with several bugs; see [Corrections](#corrections-v1--v2).

Simulated-village results are synthetic; real-village results use a year of measured data from one site. Both are directional, not field-validated.

---

## The notebook series

The notebooks in `notebooks/` tell the story in order. Each one is readable without the code.

| # | Notebook | What it shows |
|---|---|---|
| 01 | `01_the_grid` | Supply vs demand over a day, what each dispatch action physically does, and a day without diesel |
| 02 | `02_the_rulebook` | The rule-based operator and its 20-day benchmark |
| 03 | `03_the_reward_hack` | PPO trained on the original reward learns to cut households off every evening |
| 04 | `04_fixing_the_score` | PPO retrained on a corrected reward; a fair 13% saving on single days, and where it comes from |
| 05 | `05_the_morning_after` | Battery carried over between days: the single-day agent fails before dawn |
| 06 | `06_living_with_tomorrow` | PPO trained on continuous weeks; the final, honest result |
| 07 | `07_a_real_village` | A year of real demand and solar data from Agoro, Uganda; an operator rule from real practice; the AI raced on 13 unseen weeks |

---

## Results

Steady-state days from 4 continuous weeks (battery carried over between days; first day of each week excluded), 24 days per operator:

| Operator | Fuel cost/day | Unmet demand | Diesel hours | Reward |
|---|---|---|---|---|
| Rulebook | $39.30 | 0.31% | 47.4% | −42.10 |
| PPO, trained on single days | $29.95 | **5.94%** | 28.0% | −77.77 |
| **PPO, trained on weeks** | **$36.86** | **0.36%** | 42.5% | **−40.31** |

Unmet demand counts energy (kWh not delivered, including deliberately shed load) as a share of kWh demanded.

The single-day agent looks cheaper, but only because it leaves about 6% of demand unserved between 2 am and 8 am: it learned a schedule that assumes every night starts with a 70% battery, as every training day did. The week-trained agent saves fuel mainly in the morning: it lets the battery carry the quiet pre-dawn hours, arriving at sunrise with room to store more solar.

---

## Real village results (v3)

**Data:** a full year of hourly demand from Agoro, a mini-grid village in northern Uganda (homes plus productive use such as grain mills, about 105 kWh/day, peaking at 23.6 kW), and hourly PVGIS solar output for the same district. See [Data credits](#data-credits).

**System:** sized by scaling the design ratios of a commercial solar microgrid in Nigeria (about 1.5 kWp of solar per kW of peak demand, and 1.5 kWh of battery per kWp): 36 kWp solar, a 54 kWh LFP battery, and a 25 kW generator able to carry the full peak.

**Operator rule** (from real site practice): start the generator below 30% battery, charge to 50%, then stop; solar serves the load whenever the sun is up.

**Fair test:** the year is split into 52 weeks; every 4th week (13 in total) is held out. The AI trains on the other 39 and is judged only on the held-out weeks, with the battery carried over between days.

| 13 held-out weeks | Fuel cost/day | Unmet demand |
|---|---|---|
| Original rulebook | $8.28 | 0% |
| Operator rule | $6.27 | 0.13% |
| PPO, outages priced at $2/kWh | $4.16 | 0.62% |
| **PPO, outages priced at $10/kWh** | **$6.10** | **0.06%** |

At $2 per unserved kWh, the agent buys its fuel saving with outages: at dawn, leaving about 1 kWh unserved cost $2.00, while starting the generator cost $2.05. Priced at $10, above the worst-case cost of one generator hour, outages fall by 90% and the fuel saving disappears. Day by day, the agent was cheaper on 44 of 78 test days, which is not a significant difference.

Separately, about 70 kWh of solar is wasted per day in Agoro because the battery fills by early afternoon. That is a design and demand-timing limit no dispatch policy can fix.

---

## System

| Component | Specification |
|---|---|
| PV array | 50 kWp, performance ratio 0.80 → 40 kW peak at 13:00, ±15% noise; ≈ 295 kWh/day |
| Battery | 100 kWh LFP, usable 20–95% SOC (75 kWh), 30 kW charge / 40 kW discharge, round-trip η 0.92 |
| Diesel generator | 30 kW rated, 30% minimum load, 0.30 L/kWh, $0.60/L, $0.50 per start, $0.20/h when running |
| Load | 8 kW base, 35 kW peak at 20:00 (Gaussian, wraps around midnight), ±12% noise; ≈ 396 kWh/day |
| Timestep | 1 hour; episode = 24 steps (one day) or 168 steps (one week, `WeekLongEnv`) |

Note that the generator alone (30 kW) cannot cover the 35 kW evening peak; the battery has to help.

## MDP formulation

**State** (7-dim): `[SOC, PV_kW, load_kW, diesel_on, hour_sin, hour_cos, battery_health]`

**Actions** (discrete, 5):

| Index | Action | What it actually does |
|---|---|---|
| 0 | `charge_from_diesel` | Diesel serves load; surplus charges the battery |
| 1 | `discharge_battery` | PV + battery serve load; diesel off |
| 2 | `run_diesel_only` | Diesel serves load; battery covers any gap above 30 kW, and absorbs surplus |
| 3 | `solar_only` | PV serves load; battery absorbs surplus or covers the deficit (at night, this is battery operation) |
| 4 | `shed_load` | Households cut to 8 kW essential load |

**Reward** (per hour, all in USD):

```
r = − fuel − startup − fixed − VoLL × (unmet + shed) − battery_stress − daylight_diesel_penalty
```

- **VoLL** = $2.00/kWh in the simulated village and $10.00/kWh in Agoro, applied to both unmet and deliberately shed demand
- **Daylight diesel penalty**: when diesel runs while PV > 5 kW, the available PV energy is shadow-priced at the fuel rate

---

## Corrections (v1 → v2)

The originally published version is preserved at tag [`v1-published`](../../tree/v1-published). Its results came from a simulation with the following problems, all fixed in v2 with regression tests where applicable:

**Physics**
1. **PV peak was 7.2 kW, not 40 kW.** Panel efficiency was multiplied into an already-rated kWp figure. Solar covered about 14% of demand, which is why the rulebook ran diesel 62% of hours.
2. **Phantom battery energy.** When load exceeded the generator's 30 kW, both diesel actions drew the gap from the battery without checking it had any charge. At the evening peak with an empty battery, about 5.9 kWh/day appeared from nowhere for the PPO agent, recorded as served demand.
3. **Evening load ended at midnight.** The load curve didn't wrap around the clock, so demand dropped from about 24 kW to 8 kW at 00:00.

**Reward accounting**
4. **Generator start-up cost** was charged every hour the generator ran, not once per start.
5. **README/config mismatch.** The documented $0.10/h solar-hour penalty was never used by the code; the code shadow-prices PV instead. The unused key was removed.

**Evaluation**
6. **One-day episodes reset the battery to 70% every midnight**, so any energy an operator drained by the end of the day was never paid back. This flattered the PPO agent, whose learned schedule depended on it (see notebooks 05–06).
7. **"26% less diesel" measured hours, not fuel.** Generator hours and fuel burned are not proportional; fuel cost is the comparison used in v2.

**Packaging**
8. `pyproject.toml` named a non-existent build backend, could not resolve package discovery, and did not declare matplotlib.

---

## Known limitations

- **One real village.** Agoro's demand and solar come from different years (solar 2019; demand undated). Demand units (kW) and time zone are assumed, pending confirmation from the data's authors.
- **Single generator.** The reference site switches several generators on as needed; the model uses one, and assumes fuel always arrives.
- **PV curve tails.** The Gaussian solar curve gives about 0.4–2.6 kW between 4 and 6 am, before a real sunrise (≈ 4 kWh/day).
- **Wasted solar is under-logged.** Under `discharge_battery` and `shed_load`, surplus PV is neither stored nor recorded as curtailed. This does not affect costs (wasted solar is unpriced) but undercounts curtailment.
- **Linear fuel model.** Fuel use is proportional to output; real generators are less efficient at part load.
- **Simple battery degradation.** Health declines with depth of discharge only; no calendar or temperature ageing.
- **Small evaluation.** One training seed per agent; 24 test days drawn from 4 weeks, which are not fully independent.
- **Battery sizing.** Both operators waste a quarter to a third of daily solar because the battery fills by early afternoon. That is a sizing limit no dispatch policy can fix.

## Roadmap

- Operator interviews to decide the next focus: battery wear, wasted solar, or outages from fuel supply
- Priority-based load shedding (e.g. keep two of four feeders on) instead of all-or-nothing
- A Nigerian site with real operator data
- Multiple training seeds and confidence intervals
- Part-load generator efficiency curve
- Joint sizing and dispatch: how much of the remaining waste is fixed by a larger battery

---

## Project structure

```
minigrid_mdp/
├── env/
│   ├── minigrid_env.py      # Gymnasium environment (v2 physics and reward)
│   ├── week_env.py          # Same environment, one-week episodes
│   ├── data_env.py          # Real hourly solar and demand data; train/test week split
│   └── legacy_reward.py     # Original v1 reward, used only to reproduce the hack (notebook 03)
├── agents/                  # Rulebooks (original and operator practice), random, and DRL agents
├── config/
│   ├── system_params.yaml        # Simulated village
│   └── system_params_agoro.yaml  # Agoro system and reliability-first reward
├── real_data.py             # Loads PVGIS solar (UTC → local time) and demand series
├── metrics.py               # Multi-day runs and summary metrics
├── plotting.py              # One-day plot helper
├── training.py              # PPO training with the original hyperparameters
├── runner.py                # run_episode() utility
├── notebooks_v1/            # Original v1 notebook, kept for reference
├── logs/                    # v1 logs, kept for reference
└── tests/

notebooks/                   # The v2 notebook series (01–07)
models/
├── ppo_minigrid.zip         # v1 model (published results)
├── ppo_v1_reward.zip        # Original reward, corrected physics (notebook 03)
├── ppo_fixed_reward.zip     # Corrected reward, one-day episodes (notebook 04)
├── ppo_week_trained.zip     # Corrected reward, one-week episodes (notebook 06)
├── ppo_agoro_week.zip           # Agoro, outages at $2/kWh
└── ppo_agoro_week_reliable.zip  # Agoro, outages at $10/kWh

train_ppo.py                 # v1 training script; writes to ppo_minigrid.zip

data/raw/uganda_agoro/       # Agoro demand and solar data, with source note
```

## Quickstart

Requires Python 3.11+ (developed on 3.14).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,drl]"
pytest -q
```

Then open the notebooks in order. Trained models are included, so no notebook needs to retrain (training one agent takes about 8–9 minutes on CPU).

## How this was built

The v1 simulation and agents were built iteratively with Claude Code. The v2 review, corrections and notebook series were done step by step with Claude, checking each result against the code and the logs before accepting it.

## References

The parameter citations in the original version could not all be verified and have been withdrawn. Parameter values should be treated as illustrative until replaced with verified sources in the next version.

## Data credits

The Agoro demand and Lamwo solar data come from SEL-Columbia's [mini-grid_generation_system_design](https://github.com/SEL-Columbia/mini-grid_generation_system_design) repository, accompanying Wu et al., "Leveraging load shedding and daytime loads to improve mini-grid economics", *Energy Policy*. Used with the authors' permission.