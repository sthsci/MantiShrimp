# Scientific model contract

This document records the rules implemented by `mantishrimp.simulation`. It is
intended to keep model changes explicit and reviewable.

## Scope and units

The simulation is off-lattice and two-dimensional. It contains motile killer
immune cells and target cells in either a periodic rectangular domain or a
reflecting confined domain. All spatial and temporal quantities use one
user-chosen, internally consistent unit system. The reference defaults follow
the micrometre/minute interpretation used in Szonja Skenderovic's thesis.

The package implements the refined Hookean model as its maintained core. The
older Lennard–Jones scripts remain only as historical material.

## State

Each living cell has a position and an Ornstein–Uhlenbeck polarity. Killer
cells additionally have a cytotoxic state in `[0, 1]` and a productive-synapse
probability. Targets additionally have susceptibility and cumulative death
factor. A killer–target pair can be in three increasingly specific states:

1. outside proximity;
2. in physical contact;
3. bound in a synapse, which may be productive or non-productive.

The event table records transitions between these states rather than counting
every simulation frame as a new interaction.

## Motility

For an alive cell with polarity `p`,

```text
dp = -gamma * p * dt + sigma_p * sqrt(dt) * Normal(0, I)
dx = speed * p * dt + force * dt + sigma_x * sqrt(dt) * Normal(0, I)
```

Killer and target populations have separate decay, noise, and speed
parameters. Periodic calculations use minimum-image displacements. Confined
boundaries reflect both position and the corresponding polarity component.

## Mechanics and binding

For centre distance `r > 0`, let `L = R_i + R_j` and let `unit_vector` point
from cell j to cell i. All living cell pairs have the repulsive contribution

```text
F_repulsion = k_pair * max(L - r, 0) * unit_vector
```

using separate killer–killer, target–target, and killer–target stiffnesses.
A bound killer–target pair also has a constant attractive contribution,
including while the cells overlap:

```text
F_adhesion = -F_A * unit_vector    if bound and r < Rcap, otherwise 0
F_pair = F_repulsion + F_adhesion
```

The forces on the two cells are equal and opposite. In overlap, the signed
bound force is `k_pair * (L - r) - F_A`; the unbound force is
`k_pair * (L - r)`. Beyond overlap, bound pairs attract with magnitude `F_A`
until the capture cutoff; non-overlapping unbound pairs have no pair force.
Thus a bound pair can have an equilibrium at `r = L - F_A/k_pair` when that
distance is positive. An overlapping bound pair can attract or repel depending
on which contribution is stronger. Exactly coincident centres retain the
existing zero-force guard because their separation direction is undefined.

This additive rule changes the v0.1.0 behaviour, which suppressed adhesion
during overlap. Previously generated trajectories and inferred results should
remain associated with their original software version.

An unbound pair in proximity binds with probability
`1 - exp(-k_bind * dt)`. A bound pair unbinds with probability
`1 - exp(-k_unbind * dt)` and also separates deterministically beyond the
capture radius. These rate-to-probability conversions make the stochastic
rules consistent under changes to the integration step.
The default formation threshold is `L + epsilon = 1.05L`, while
`Rcap = 1.5L` limits retention of an existing bond. The force is zero at
`r = Rcap`; deterministic bond removal uses `r > Rcap`. Binding is restricted
to living killer–target pairs, and target death also ends the synapse.

## Cytotoxicity

When a synapse forms, its productive state is sampled once from the killer's
current killing probability. The reference homogeneous model sets this
probability to one. Optional variation and per-synapse decay support later
extensions without conflating binding with killing.

Each productive bound killer contributes to target damage at

```text
target_susceptibility * (
    killing_rate_min
    + (killing_rate_max - killing_rate_min) * killer_state
)
```

Target damage recovers at a first-order rate. A target dies once damage reaches
the configured threshold. The kill event is assigned to the killer with the
largest cumulative damage contribution, while all contributors remain
recoverable from synapse history. Killer state decreases with productive
synapse time according to target susceptibility and the configured exhaustion
rates.

## Integration order

Each step computes an adaptive `dt`, updates binding/unbinding, recomputes
forces, updates polarity and position, applies boundaries, integrates damage
and exhaustion, resolves death/separation, then records contact transitions.
The adaptive step includes a conservative bound on force changes from possible
binding/unbinding events. This prevents a transition out of a force-balanced
bound state from exceeding the deterministic displacement limit. The final
`dt` is used for the transition probabilities and never exceeds `max_dt` or
the remaining duration. Translational noise is not bounded by this drift limit.

## Inference observables

The Bayesian layer consumes event-episode counts per killer:

- contacts: `contact_started` events;
- synapses: `synapse_formed` events;
- kills: primary-attributed `target_killed` events.

Every killer is retained, including cells with zero events. The default
exposure is the simulation duration. This is a population-level count model,
not a full likelihood for trajectories or dependent contact networks; that
assumption should be tested when calibrating against experimental data.
