# MantiShrimp

[![CI](https://github.com/sthsci/MantiShrimp/actions/workflows/ci.yml/badge.svg)](https://github.com/sthsci/MantiShrimp/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sthsci/MantiShrimp/blob/main/LICENSE)

MantiShrimp is a Python package for agent-based modelling of killer immune
cell–tumour cell interactions and Bayesian analysis of the resulting
cell-level event counts.

The package has two connected layers:

- an off-lattice, two-dimensional ABM using the refined Hookean interaction
  rules from Szonja Skenderovic's thesis;
- the four Bayesian count models from the Orca inference work, applicable to
  either contacts per killer cell or kills per killer cell.

The simulation engine lives in `src/mantishrimp`. The existing
`script/simulator/simulator.py` entry point calls that same engine and keeps
the `KTSimulator` arguments and six returned outputs used by older notebooks.

## ABM demonstration

![Animated killer-target ABM simulation](https://raw.githubusercontent.com/sthsci/MantiShrimp/main/figures/abm_demo.gif)

Blue agents are killer immune cells, green agents are living tumour targets,
and salmon agents are dead targets. Cells move continuously in the off-lattice
domain; killer-target proximity can lead to stochastic synapse formation,
damage accumulation, and target death. The animation is a visual illustration
of one run rather than a calibrated biological prediction. A
[full-resolution version](https://github.com/sthsci/MantiShrimp/blob/main/figures/vis_test.gif)
is also available.

## Install

```bash
python -m pip install mantishrimp
python -m pip install 'mantishrimp[inference]'
python -m pip install 'mantishrimp[all]'  # inference and plotting
```

Python 3.10 or newer is supported.

## Simulate killer–target interactions

```python
from mantishrimp import SimulationConfig, simulate

config = SimulationConfig.szonja_baseline(
    n_killers=25,
    n_targets=100,
    duration=25.0,  # shortened from the 80-minute thesis baseline
    seed=42,
)
result = simulate(config)

print(result.summary())
print(result.contacts_per_cell())
print(result.kills_per_cell())
```

`SimulationResult` contains:

- `snapshots`: one row per recorded cell and time point;
- `events`: contact, synapse, and death events with killer/target identities;
- `final_cells`: the final recorded population state;
- `config`: the complete typed simulation configuration.

Contacts and synapses are deliberately different. A contact is a proximity
episode; a synapse is a stochastic bound state formed during contact. This
keeps the biological mechanism separate from the observable used for
inference.

### Use an existing simulation script or notebook

Keep using `from simulator import KTSimulator` when your notebook already adds
`script/simulator` to its Python path. From the repository root, you can also use:

```python
from script.simulator.simulator import KTSimulator

history, killers, targets, probabilities, damage, settings = KTSimulator(
    N_KILLER=10, N_TARGET=40, SIM_DURATION=1.0, RNG_SEED=42, ini_seed=42,
)
```

Both entry points use the same movement, binding, death and exhaustion rules.
Dead targets retain existing bonds until unbinding or capture-radius separation;
productive killers continue to exhaust while bound. Released dead targets cannot
bind again or affect unbound cells.

The legacy arguments retain their defaults and spatial initialisers, so matching
trajectories requires matching parameters, initial cell states and positions,
and the dynamics seed. `RECOVERY=True` uses cumulative damage; `RECOVERY=False`
preserves the legacy instant-kill mode. To use the package interface directly
from the simulator folder, import `SimulationConfig` and `simulate` from
`simulator`; these are the package objects themselves.

Keep the repository folders together; copying only `simulator.py` is insufficient.
Restart the notebook kernel after updating the code. Edit model rules in
`src/mantishrimp/simulation.py`; both entry points will use those changes.

<!--
## Infer heterogeneity from ABM results

The inference layer asks a population-level question: **are killer cells
consistent with one common event rate, or is there evidence for inactive cells,
continuous rate heterogeneity, or both?** It operates on one integer count per
killer cell. That count can be either the number of contact episodes or the
number of attributed kills.

![Synthetic event-count and Bayesian inference workflow](https://raw.githubusercontent.com/sthsci/MantiShrimp/main/figures/syn_vali.png)

The workflow above has three steps:

1. Each killer cell `i` has a latent event rate `λᵢ`.
2. Over exposure time `Tᵢ`, its observed count is modelled as
   `Nᵢ | λᵢ, Tᵢ ~ Poisson(λᵢTᵢ)`.
3. Bayesian inference combines the count likelihood with the parameter priors
   to estimate `p(θ | D)`, where `θ = (μλ, σλ, φ₀)`.

Here `μλ` is the mean event rate among active killer cells, `σλ` is their
between-cell rate variation, and `φ₀` is the
fraction of structurally inactive killer cells. Zero-count cells are retained:
an observed zero can arise either from an active Poisson process that happened
to produce no events or, in zero-inflated models, from the inactive component.
The default exposure is the simulated duration, although aligned per-cell
exposures can be supplied directly.

### Four candidate population models

![Four Bayesian event-rate population models](https://raw.githubusercontent.com/sthsci/MantiShrimp/main/figures/models.png)

The figure expresses the four models as restrictions of the same latent-rate
distribution. The grey/blue mass at zero is `φ₀`; the positive-rate
distribution has mean `μλ` and standard deviation `σλ`.

| Package name | Figure | Parameter restriction | Count distribution | Interpretation |
| --- | --- | --- | --- | --- |
| `homo` | homo | `σλ = 0; φ₀ = 0` | Poisson | one shared rate |
| `Z2P` | ZI | `σλ = 0; φ₀ > 0` | zero-inflated Poisson | shared active rate plus inactive cells |
| `Dis2P` | Γ | `σλ > 0; φ₀ = 0` | negative binomial | Gamma-distributed active-cell rates |
| `hetero3` | ZI Γ | `σλ > 0; φ₀ > 0` | zero-inflated negative binomial | Gamma rates plus inactive cells |

For the Gamma models, integrating the cell-specific `λᵢ` values out of
the Poisson likelihood gives the negative-binomial count distribution. This
lets the model estimate continuous cell-to-cell variation without sampling a
separate rate for every killer.

The original vector figures are available as
[the inference workflow PDF](https://github.com/sthsci/MantiShrimp/blob/main/figures/syn_vali.pdf)
and
[the four-model PDF](https://github.com/sthsci/MantiShrimp/blob/main/figures/models.pdf).

### Fit contacts and kills separately

Install the `inference` extra, then fit the same candidate suite to each
observable:

```python
from mantishrimp.inference import evidence_table, infer_contacts, infer_kills

contact_fits = infer_contacts(
    result,
    draws=1000,
    chains=2,
    random_seed=42,
)
kill_fits = infer_kills(
    result,
    draws=1000,
    chains=2,
    random_seed=43,
)

print(evidence_table(contact_fits))
print(evidence_table(kill_fits))
```

Contacts and kills should be analysed as separate datasets because they answer
different biological questions: contact inference characterises encounter
opportunity, whereas kill inference combines encounter, synapse, damage, and
cytotoxic competence.

### Posterior recovery and model comparison

Models are fitted with PyMC Sequential Monte Carlo (SMC). Each `FitResult`
contains posterior samples for parameter recovery and a log marginal
likelihood for model comparison. With equal prior model probabilities,

```text
BF(A, B) = p(D | M_A) / p(D | M_B) = exp(log Z_A - log Z_B)
```

The posterior answers *which parameter values are plausible within a model*;
the Bayes factor answers *which population model is better supported by the
observed counts*. `evidence_table`, `bayes_factor_matrix`, and the plotting
helpers expose both views. Raw count arrays can also be fitted with
`fit_count_model` or `fit_model_suite`.

This count likelihood is an observation model for ABM outputs; it does not by
itself infer Hookean force constants, motility parameters, or causal cell-cell
network structure. Those would require a separate calibration or
simulation-based inference layer.
-->

## Package map

- `config.py`: validated simulation parameters and the Szonja baseline;
- `simulation.py`: Hookean ABM engine;
- `results.py`: event-count extraction and result persistence;
- `inference.py`: Orca-derived Bayesian model suite;
- `validation.py`: synthetic Poisson/Gamma/zero-inflated generators;
- `analysis.py`: evidence, posterior, sweep, and persistence helpers;
- `plotting.py`: reusable posterior and Bayes-factor plots.

See the [scientific model contract](https://github.com/sthsci/MantiShrimp/blob/main/docs/model.md)
and [notebook-function migration map](https://github.com/sthsci/MantiShrimp/blob/main/docs/notebook-migration.md).

## Authors and citation

MantiShrimp was developed by Elephes Sung, Szonja Skenderovic, Yixuan Li, and
Ruben Perez-Carrasco at the Department of Life Sciences, Imperial College
London. Elephes Sung and Szonja Skenderovic contributed equally. See
[`CITATION.cff`](https://github.com/sthsci/MantiShrimp/blob/main/CITATION.cff)
for author details and machine-readable citation metadata.

## Development

The `tests/` directory is kept locally and is not included in GitHub checkouts.
Run `pytest` below only when those local test files are available.

```bash
python -m pip install -e '.[all,test]'
pytest
```

The package is currently alpha software. Before using it for biological
claims, calibrate the time/space units, parameter priors, and observation
process against the experiment being modelled.
