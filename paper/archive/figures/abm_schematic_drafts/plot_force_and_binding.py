"""Plot the implemented pair force and its separate synapse-state rules.

Run with the existing abm environment. Amplitudes are illustrative, not the
baseline calibration. The curve is evaluated by the simulator itself.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mantishrimp.config import MechanicsConfig, SimulationConfig
from mantishrimp.simulation import _forces

OUT = Path(__file__).resolve().parent
config = SimulationConfig(
    n_killers=1, n_targets=1, killer_radius=0.5, target_radius=0.5,
    mechanics=MechanicsConfig(repulsion_kt=1.0, adhesion_kt=0.35),
)


def force(distance, bound):
    positions = np.array([[distance, 0.0], [0.0, 0.0]])
    return _forces(positions, np.array([True, True]),
                   np.array([[bound]]), config)[0, 0]


# Bound adhesion subtracts F_A from the overlap force as well.
assert np.isclose(force(0.8, False), 0.2)
assert np.isclose(force(0.8, True), -0.15)
assert np.isclose(force(0.65, True), 0.0)
assert force(1.2, False) == 0 and np.isclose(force(1.2, True), -0.35)
assert force(1.5, True) == 0 and force(1.6, True) == 0

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "svg.fonttype": "none",
})
bound_color, unbound_color = "#84364a", "#656565"
fig = plt.figure(figsize=(12.5, 5.4), facecolor="white")
ax = fig.add_axes([0.075, 0.28, 0.415, 0.54])
ax.axvspan(0, 1, color="#f4e3df", zorder=0)
ax.axvspan(1, 1.5, color="#e5e9f3", zorder=0)
ax.axhline(0, color="#303030", lw=0.9, zorder=1)
for cutoff in [1, 1.5]:
    ax.axvline(cutoff, color="#a0a0a0", lw=0.9, ls=":")
for state, color, ls, label in [
    (True, bound_color, "-", r"bound, $B_{kt}=1$"),
    (False, unbound_color, "--", r"unbound, $B_{kt}=0$"),
]:
    # Separate segments preserve discontinuities instead of interpolating them.
    for i, (start, stop) in enumerate([(0.025, 1-1e-7), (1, 1.5-1e-7), (1.5, 1.92)]):
        x = np.linspace(start, stop, 100)
        ax.plot(x, [force(d, state) for d in x], color=color, ls=ls,
                lw=2.7, label=label if i == 0 else None)
ax.plot([1.5, 1.5], [-0.35, 0], ":", color=bound_color, lw=1.3)
ax.plot(1, -0.35, "o", color=bound_color, ms=5)
ax.plot(1.5, -0.35, "o", mfc="white", mec=bound_color, ms=5, zorder=5)
ax.plot(1.5, 0, "o", color=bound_color, ms=5, zorder=5)
ax.set(xlim=(0, 1.95), ylim=(-0.60, 1.12), xlabel=r"centre distance $\delta$",
       ylabel=r"signed pair force $F(\delta,B_{kt})$")
ax.set_xticks([1, 1.5], [r"$L$", r"$R_{\mathrm{cap}}$"])
ax.set_yticks([-0.35, 0, 1], [r"$-F_A$", "0", r"$K_{\mathrm{rep}}L$"])
for x, text in [(0.48, "overlap"), (1.25, "adhesion\nif bound"), (1.73, "no pair\nforce")]:
    ax.text(x, 1.015, text, ha="center", va="center", fontsize=11)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.80), fontsize=10)
fig.text(0.075, 0.91, "Mechanical response", fontsize=18)
fig.text(0.075, 0.855, "Positive = repulsion; negative = attraction", fontsize=11)

state_ax = fig.add_axes([0.56, 0.23, 0.41, 0.59])
state_ax.set(xlim=(0, 1), ylim=(0, 1))
state_ax.axis("off")
fig.text(0.56, 0.91, "Synapse state", fontsize=18)
fig.text(0.56, 0.855, "Each living killer–target pair has its own binding state", fontsize=11)
for x, label in [(0.02, "unbound\n$B_{kt}=0$"), (0.76, "bound\n$B_{kt}=1$")]:
    state_ax.add_patch(FancyBboxPatch((x, 0.39), 0.22, 0.22,
                       boxstyle="round,pad=0.012,rounding_size=0.025",
                       facecolor="white", edgecolor="#303030", lw=1.4))
    state_ax.text(x+0.11, 0.50, label, ha="center", va="center", fontsize=14)
state_ax.annotate("", xy=(0.745, 0.58), xytext=(0.255, 0.58),
                  arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#303030"))
state_ax.text(0.50, 0.90, "BIND only near contact", ha="center", fontsize=12)
state_ax.text(0.50, 0.79, r"$\delta\leq L+\epsilon$", ha="center", fontsize=15)
state_ax.text(0.50, 0.68, r"$p_{\mathrm{bind}}=1-e^{-k_{\mathrm{bind}}\Delta t}$", ha="center", fontsize=14)
state_ax.annotate("", xy=(0.255, 0.43), xytext=(0.745, 0.43),
                  arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#303030"))
state_ax.text(0.50, 0.29, "UNBIND stochastically", ha="center", fontsize=12)
state_ax.text(0.50, 0.17, r"$p_{\mathrm{unbind}}=1-e^{-k_{\mathrm{unbind}}\Delta t}$", ha="center", fontsize=14)
state_ax.text(0.50, 0.015, r"Also ends if $\delta>R_{\mathrm{cap}}$ or the target dies", ha="center", fontsize=11)
fig.text(0.56, 0.13, r"$L+\epsilon=1.05L$: can form a new bond", fontsize=12)
fig.text(0.56, 0.075, r"$R_{\mathrm{cap}}=1.5L$: can retain an existing bond", fontsize=12)
fig.text(0.075, 0.13, "The force curve does not decide whether a bond forms.", fontsize=11)
fig.text(0.075, 0.075, "Force amplitudes are schematic, not baseline parameter values.", fontsize=10, color="#555555")

for extension in ["svg", "png"]:
    path = OUT / f"force_and_binding_rules_additive.{extension}"
    fig.savefig(path, dpi=300, facecolor="white")
    print(path)
