import matplotlib.pyplot as plt


def plot_day(day, initial_soc, min_soc, title=None):
    fig, (ax_power, ax_soc) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_power.plot(day.hour, day.pv_kw, label="Solar", color="orange")
    ax_power.plot(day.hour, day.load_kw, label="Demand", color="steelblue")
    ax_power.step(day.hour, day.diesel_kw, where="mid", label="Diesel", color="dimgray")
    ax_power.bar(day.hour, day.unmet_load_kw, label="Unmet demand", color="crimson", alpha=0.5)
    ax_power.set_ylabel("kW")
    ax_power.legend()

    soc_curve = [initial_soc] + day.soc_after.tolist()
    ax_soc.plot(range(len(soc_curve)), soc_curve, color="seagreen", marker="o", markersize=3)
    ax_soc.axhline(min_soc, color="gray", linestyle="--", label="20% floor")
    ax_soc.set_ylabel("Battery SOC")
    ax_soc.set_xlabel("Hour of day")
    ax_soc.set_xticks(range(0, 25, 3))
    ax_soc.legend()

    if title:
        fig.suptitle(title)
    return fig