"""Editable panel c: illustrative binding schedules, default damage rules.

The contact schedules are prescribed to explain the rules, not sampled ABM
outputs. Damage and exhaustion follow the update order in simulation.py.
No original image is modified by this script.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mantishrimp.config import CytotoxicityConfig

OUT = Path(__file__).resolve().parent
CYTO = CytotoxicityConfig()


def illustrative_trace(intervals, end=3.5, dt=0.001):
    """One killer and one target; productive binding is prescribed."""
    t, damage, state = 0.0, 0.0, CYTO.initial_killer_state
    susceptibility = CYTO.initial_target_state
    times, values = [t], [damage]
    while t < end - 1e-10:
        step = min(dt, end - t)
        bound = any(start <= t + 1e-9 < stop for start, stop in intervals)
        if bound:
            rate = susceptibility * (
                CYTO.killing_rate_min
                + (CYTO.killing_rate_max - CYTO.killing_rate_min) * state
            )
            damage += rate * step
        damage -= CYTO.recovery_rate * damage * step
        damage = max(0.0, damage)
        if bound:
            exhaustion = CYTO.exhaustion_rate_min + (
                CYTO.exhaustion_rate_max - CYTO.exhaustion_rate_min
            ) * susceptibility
            state *= np.clip(1 - exhaustion * step, 0, 1)
        t += step
        times.append(t)
        values.append(min(damage, CYTO.death_threshold) / CYTO.death_threshold)
        if damage >= CYTO.death_threshold:
            break
    return np.array(times), np.array(values)


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 11, "axes.linewidth": 1.0,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.8, 2.95), facecolor="white")
    green, pink, orange = "#268538", "#ec6579", "#ea933e"
    fig.text(0.013, 0.91, "c", fontsize=23)
    fig.text(0.088, 0.91, "Target death-factor dynamics", fontsize=16)
    fig.patches.append(Rectangle((0.675, 0.914), 0.019, 0.044,
                                 transform=fig.transFigure,
                                 facecolor=orange, alpha=0.15, edgecolor="none"))
    fig.text(0.703, 0.915, "productive synapse (bound)", fontsize=11)

    plans = [
        (0.089, [(0.35, 0.90), (1.70, 3.5)], "Repeated binding", True),
        (0.565, [(0.35, 0.90)], "Short binding", False),
    ]
    for x, intervals, title, dies in plans:
        ax = fig.add_axes([x, 0.235, 0.387, 0.51])
        t, d = illustrative_trace(intervals)
        final_t = t[-1]
        for start, stop in intervals:
            if start < final_t:
                ax.axvspan(start, min(stop, final_t), color=orange,
                           alpha=0.13, lw=0, zorder=0)
        ax.axhline(1, ls=(0, (3, 3)), color="#777777", lw=1.0, zorder=1)
        ax.plot(t, d, color=green, lw=2.2, solid_capstyle="round", zorder=3)
        ax.set(xlim=(0, 3.5), ylim=(0, 1.14))
        ax.set_xticks([])
        ax.set_yticks([0, 1], ["0", "1"] if dies else ["", ""])
        ax.tick_params(axis="y", length=3, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, 1.23, title, ha="left", fontsize=12)
        ax.set_xlabel("time", labelpad=5)
        ax.text(3.49, 1.025, "death threshold", ha="right", va="bottom",
                color="#555555", fontsize=10)
        ax.annotate("", xy=(3.57, 0), xytext=(3.42, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1),
                    annotation_clip=False)
        if dies:
            ax.set_ylabel(r"death factor $D/D_{\mathrm{crit}}$", labelpad=5)
            ax.scatter([final_t], [1], s=40, color=pink, edgecolor="white",
                       linewidth=0.7, zorder=5)
            ax.text(final_t + 0.10, 0.86, "target death", color=pink,
                    va="top", fontsize=11)
            ax.text(1.30, 0.32, "unbound", ha="center", color="#777777", fontsize=10)
            ax.annotate("", xy=(1.30, d[np.searchsorted(t, 1.30)] - 0.035),
                        xytext=(1.30, 0.44),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#999999"))
        else:
            ax.scatter([final_t], [d[-1]], s=28, color=green,
                       edgecolor="white", linewidth=0.7, zorder=5,
                       clip_on=False)
            ax.text(2.1, d[-1] + 0.09, "alive at end", color=green,
                    ha="center", fontsize=11)
            ax.text(2.1, d[-1] - 0.14, "damage retained", color="#777777",
                    ha="center", fontsize=10)

    fig.text(0.089, 0.045, "No recovery (default): damage is retained between binding episodes.",
             fontsize=10.5, color="#555555")
    for ext in ("svg", "pdf", "png"):
        path = OUT / f"panel_c_death_factor_default.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
