import pandas as pd

from minigrid_mdp.runner import run_episode

ACTION_NAMES = ["charge_from_diesel", "discharge_battery", "run_diesel_only", "solar_only", "shed_load"]


def run_days(env, agent, n_days=20):
    return pd.concat([run_episode(env, agent, episode_idx=ep) for ep in range(n_days)], ignore_index=True)


def summarise(log):
    shed_kw = log.curtailed_kw.where(log.action == 4, 0.0)
    log = log.assign(unserved_kw=log.unmet_load_kw + shed_kw)
    per_day = log.groupby("episode")
    return pd.DataFrame({
        "diesel_hours_pct": per_day.diesel_on.mean() * 100,
        "fuel_cost_usd":    -per_day.rew_fuel.sum(),
        "unmet_pct":        per_day.unserved_kw.sum() / per_day.load_kw.sum() * 100,
        "end_soc":          per_day.soc_after.last(),
        "reward":           per_day.reward.sum(),
    })


def compare(results):
    return pd.concat({name: df.agg(["mean", "std"]).T for name, df in results.items()}, axis=1).round(2)


def action_share_by_hour(log):
    shares = pd.crosstab(log.hour, log.action, normalize="index")
    return shares.rename(columns=dict(enumerate(ACTION_NAMES))).round(2)

def run_weeks(env, agent, n_weeks=4, days_per_week=7):
    log = run_days(env, agent, n_weeks)
    return log.assign(episode=log.episode * days_per_week + log.step // 24)


def steady_state_days(log, days_per_week=7):
    return log[log.episode % days_per_week != 0]