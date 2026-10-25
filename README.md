# Mini-Grid Dispatch RL

A Gymnasium simulation of a **50 kW solar–battery–diesel mini-grid** serving a rural Nigerian village, and an honest test of whether a reinforcement-learning agent (PPO) can operate it better than a fixed rulebook.

**Short answer:** yes, but by much less than first reported. On continuous multi-week runs, a PPO agent trained on week-long episodes spends **about 6% less on fuel than the rulebook at the same reliability**, and is cheaper on 19 of 24 test days. The originally published figure (26% fewer diesel hours) came from a simulation with several bugs; see [Corrections](#corrections-v1--v2).

All data is synthetic. Results are directional, not field-validated.

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

- **VoLL** = $2.00/kWh, applied to both unmet and deliberately shed demand
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

- **Synthetic data only.** No real irradiance, load or operational data yet.
- **PV curve tails.** The Gaussian solar curve gives about 0.4–2.6 kW between 4 and 6 am, before a real sunrise (≈ 4 kWh/day).
- **Wasted solar is under-logged.** Under `discharge_battery` and `shed_load`, surplus PV is neither stored nor recorded as curtailed. This does not affect costs (wasted solar is unpriced) but undercounts curtailment.
- **Linear fuel model.** Fuel use is proportional to output; real generators are less efficient at part load.
- **Simple battery degradation.** Health declines with depth of discharge only; no calendar or temperature ageing.
- **Small evaluation.** One training seed per agent; 24 test days drawn from 4 weeks, which are not fully independent.
- **Battery sizing.** Both operators waste a quarter to a third of daily solar because the battery fills by early afternoon. That is a sizing limit no dispatch policy can fix.

## Roadmap

- Real hourly irradiance for a specific Nigerian site, and real or measured load profiles
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
│   └── legacy_reward.py     # Original v1 reward, used only to reproduce the hack (notebook 03)
├── agents/                  # Rule-based, random, and DRL (SB3 model wrapper) agents
├── config/
│   └── system_params.yaml   # Physical and reward parameters
├── metrics.py               # Multi-day runs and summary metrics
├── plotting.py              # One-day plot helper
├── training.py              # PPO training with the original hyperparameters
├── runner.py                # run_episode() utility
├── notebooks_v1/            # Original v1 notebook, kept for reference
├── logs/                    # v1 logs, kept for reference
└── tests/

notebooks/                   # The v2 notebook series (01–06)
models/
├── ppo_minigrid.zip         # v1 model (published results)
├── ppo_v1_reward.zip        # Original reward, corrected physics (notebook 03)
├── ppo_fixed_reward.zip     # Corrected reward, one-day episodes (notebook 04)
└── ppo_week_trained.zip     # Corrected reward, one-week episodes (notebook 06)
train_ppo.py                 # v1 training script; writes to ppo_minigrid.zip
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