# Changelog

All notable changes to MantiShrimp are documented in this file.

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `script/simulator/simulator.py` now delegates to the package engine. Existing
  `KTSimulator` calls receive the post-death binding and continued exhaustion
  rules while retaining their arguments and six returned outputs.
- The standalone force helper uses the package's binding and pair-force routines,
  and the three parameter sweeps import the maintained simulator entry point.

### Changed

- Shared force calculation now uses batched NumPy operations instead of Python
  loops over cell pairs. Both package and legacy script calls use this routine,
  including the assistant helper's pairwise output. The force rules, adaptive
  timestep, random draws and recording frequency are unchanged. Floating-point
  rounding can differ, and temporary pair arrays require quadratic memory.
- Legacy simulations now use package integration, periodic geometry and primary
  kill attribution. Their `contacts` lists retain dead-target bonds until release,
  and `killed_by` records one primary killer per death.
- `RNG_SEED` controls legacy population sampling and dynamics reproducibly.
  Invalid initial states, probabilities and death factors are rejected.

### Added

- Explicit initial population arrays in `Simulation.run`, snapshot synapse IDs,
  killing probabilities and primary-killer IDs for legacy output conversion.
- Shared instant-kill mode for legacy `RECOVERY=False` calls. The package default
  still uses cumulative damage.
- Regression checks comparing script and package trajectories under identical
  inputs, plus legacy post-death binding, exhaustion, initialisation and imports.
- Scalar-reference comparisons for vectorized forces and seeded dynamics,
  including both boundary conditions, both output formats and retained dead bonds.

## [0.3.0] - 2026-10-14

### Changed

- In the package ABM, target death now preserves existing synapses until
  stochastic unbinding at the configured rate or capture-radius separation.
  Dead targets remain stationary and exert pair forces on bound killers.
- Productive conjugation continues to exhaust killers after target death.
  Dead-target damage and kill attribution remain fixed, and dead targets
  cannot form new synapses.
- The adaptive timestep accounts for removal of the full pair force when
  a killer unbinds from a dead target. These model changes affect trajectories,
  synapse durations, and subsequent killing relative to v0.2.0.

### Added

- Regression coverage for retained bonds and pair forces after death,
  exhaustion until release, non-productive bonds, and the displacement limit
  when unbinding from dead targets.

## [0.2.0] - 2026-10-02

### Changed

- Bound killer–target adhesion now adds to Hookean repulsion during overlap,
  giving signed force `Krep * (L - distance) - F_A` for overlapping bound
  pairs. Both the package simulator and the legacy Hookean helper use this
  rule. Binding/unbinding rates and the capture cutoff are unchanged.
- This changes the default scientific model relative to v0.1.0; previous
  trajectories and analyses retain their original model provenance.
- The package's adaptive timestep now accounts for possible force changes on
  binding/unbinding, including release from an overlapping force equilibrium.

### Added

- Regression tests for additive bound forces, capture boundaries, and the
  displacement limit following unbinding at force equilibrium.
- A force and binding schematic evaluated directly using the simulator,
  with an editable SVG and reproduction script.

## [0.1.0] - 2026-08-12

Initial public package release.

### Added

- Off-lattice killer-target ABM with Ornstein-Uhlenbeck motility.
- Hookean overlap repulsion and explicit killer-target adhesion.
- Continuous-time stochastic synapse binding and unbinding rules.
- Cumulative target damage, recovery, death, and killer exhaustion.
- Separate proximity-contact, bound-synapse, and attributed-kill events.
- Analysis-ready `SimulationResult` with portable CSV/JSON persistence.
- Bayesian inference for contacts or kills per killer cell using homogeneous,
  zero-inflated, Gamma-Poisson, and zero-inflated Gamma-Poisson models.
- Explicit PyMC 5 and ArviZ 0.x compatibility bounds for the initial inference
  API.
- SMC marginal-likelihood extraction, posterior summaries, and Bayes factors.
- Synthetic validation, analysis, and plotting utilities extracted from Orca.
- Typed configuration, documentation, packaging metadata, and unit tests.

[Unreleased]: https://github.com/sthsci/MantiShrimp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/sthsci/MantiShrimp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sthsci/MantiShrimp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sthsci/MantiShrimp/releases/tag/v0.1.0
