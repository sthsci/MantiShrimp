"""Plot the simulator's pair force for the mechanics values discussed in the notebook."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/mantishrimp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from mantishrimp import DomainConfig, MechanicsConfig, SimulationConfig
from mantishrimp.simulation import _forces

# The displayed adhesion is the user's new example (250); notebook defaults are separate.
config = SimulationConfig(
    n_killers=1, n_targets=1, killer_radius=10, target_radius=12,
    domain=DomainConfig(x_min=-600, x_max=600, y_min=-600, y_max=600),
    mechanics=MechanicsConfig(
        repulsion_kk=250, repulsion_tt=250, repulsion_kt=250,
        adhesion_kt=250, binding_rate=5.0, unbinding_rate=0.5,
    ),
)
config.validate()
m = config.mechanics
L = config.killer_radius + config.target_radius
Rcap = m.capture_radius_factor * L
equilibrium = L - m.adhesion_kt / m.repulsion_kt


def radial_force(distance, bound):
    """Positive means the killer is pushed away from the target at the origin."""
    return _forces(
        np.array([[distance, 0.0], [0.0, 0.0]]),
        np.ones(2, dtype=bool), np.array([[bound]], dtype=bool), config,
    )[0, 0]


# Check both branches, their force balance, and the strict adhesion cutoff.
probe = np.array([0.1, 10, equilibrium, L, 25, Rcap - 1e-8, Rcap, 36])
for bound in [False, True]:
    expected = m.repulsion_kt * np.maximum(L - probe, 0)
    if bound:
        expected -= m.adhesion_kt * (probe < Rcap)
    np.testing.assert_allclose([radial_force(d, bound) for d in probe], expected)
assert radial_force(equilibrium, True) == 0
assert radial_force(L, True) == -250
assert radial_force(Rcap, True) == 0
assert radial_force(0, True) == 0  # Coincident centres have no defined radial direction.

# Keep the force jump at Rcap explicit rather than interpolating through it.
distance = np.unique(np.r_[np.linspace(0.01, 37, 650), equilibrium, L, Rcap])
unbound = np.array([radial_force(d, False) for d in distance])
below = distance[distance < Rcap]
bound_below = np.array([radial_force(d, True) for d in below])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300,
})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.80, wspace=0.27)
fig.suptitle('How repulsion and adhesion set the killer–target force', y=0.98, fontsize=16)
fig.text(0.5, 0.905, 'repulsion_kt = 250   ·   adhesion_kt = 250   ·   radii = 10 and 12', ha='center', fontsize=11)
green, orange = '#009E73', '#D55E00'

for ax in axes:
    ax.axhline(0, color='0.2', linewidth=1)
    ax.plot(distance, unbound, color=green, linewidth=2.5, linestyle='--', label='Unbound: repulsion only', zorder=3)
    ax.plot(np.r_[below, Rcap], np.r_[bound_below, -m.adhesion_kt], color=orange, linewidth=2.5,
            label='Bound: repulsion + adhesion', zorder=2)
    ax.plot([Rcap, distance[-1]], [0, 0], color=orange, linewidth=2.5, zorder=2)
    ax.plot([Rcap, Rcap], [-m.adhesion_kt, 0], color=orange, linewidth=1, linestyle=':', zorder=2)
    ax.plot(Rcap, -m.adhesion_kt, 'o', markerfacecolor='white', markeredgecolor=orange, markersize=5, zorder=5)
    ax.plot(Rcap, 0, 'o', color=orange, markersize=5, zorder=5)
    for value in [L, Rcap]:
        ax.axvline(value, color='0.65', linewidth=0.9, linestyle=':', zorder=0)
    ax.set_xlabel(r'Centre distance $\delta$ (model length units)')
    ax.set_ylabel('Signed radial force (model units)')
    ax.tick_params(direction='out')

ax = axes[0]
ax.set(title='Full force profile', xlim=(0, 37), ylim=(-650, 5900), xticks=[0, 10, 20, 30])
ax.legend(loc='upper right', fontsize=9.5)
ax.annotate('Stiffness = 250\n1 unit of overlap adds 250', xy=(9, 3250), xytext=(17, 3000),
            arrowprops={'arrowstyle': '->', 'color': '0.35'}, fontsize=10)
ax.text(1.5, 900, 'Positive: repulsion', color='0.25', fontsize=10)
ax.text(1.5, -500, 'Negative: attraction', color='0.25', fontsize=10)
ax.text(L, 1000, r'$L=22$', ha='center', color='0.25')
ax.text(Rcap, 1000, r'$R_{\mathrm{cap}}=33$', ha='center', color='0.25')

ax = axes[1]
ax.set(title='Close-up around contact', xlim=(18, 36), ylim=(-480, 1100),
       xticks=[18, 21, 24, 27, 30, 33, 36], yticks=[-250, 0, 250, 500, 750, 1000])
ax.plot(equilibrium, 0, 'o', color='0.15', markersize=5, zorder=6)
ax.annotate(r'Force balance: $\delta=21$' + '\n1 unit of overlap', xy=(equilibrium, 0), xytext=(18.4, -420),
            arrowprops={'arrowstyle': '->', 'color': '0.35'}, fontsize=10)
ax.text(L + 0.25, 970, r'Contact $L=22$', fontsize=10)
ax.text(Rcap - 0.25, 780, 'Adhesion cutoff\n' + r'$R_{\mathrm{cap}}=33$', ha='right', fontsize=10)
ax.text(27.4, -210, r'$-F_A=-250$', ha='center', va='bottom', color=orange, fontsize=11)

fig.text(0.085, 0.09, r'$L=r_K+r_T$; $R_{\mathrm{cap}}=1.5L$.  Positive force separates cells; negative force pulls them together.', fontsize=10)
fig.text(0.085, 0.045, r'Pair force for living cells and $\delta>0$, without motility or other cells. Binding rates switch the pair state.', fontsize=10, color='0.3')
output = Path(__file__).resolve().parent
for suffix in ['png', 'pdf']:
    fig.savefig(output / f'fig_mechanics_forces.{suffix}')
plt.close(fig)
print('Verified against _forces: L=22, Rcap=33, equilibrium=21, adhesion plateau=-250.')
print(output / 'fig_mechanics_forces.png')
