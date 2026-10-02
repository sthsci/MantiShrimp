"""Hookean killer-target agent-based simulation engine.

The implementation follows the refined model in Szonja Skenderovic's thesis:
off-lattice Ornstein-Uhlenbeck motility, Hookean overlap repulsion, explicit
stochastic killer-target binding/unbinding, cumulative target damage and
killer exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import DomainConfig, SimulationConfig
from .results import SimulationResult


EVENT_COLUMNS = [
    "time",
    "step",
    "event",
    "killer_id",
    "target_id",
    "value",
    "reason",
]


def _minimum_image(displacement: np.ndarray, domain: DomainConfig) -> np.ndarray:
    out = np.asarray(displacement, dtype=float).copy()
    if domain.boundary == "periodic":
        out[..., 0] -= domain.width * np.round(out[..., 0] / domain.width)
        out[..., 1] -= domain.height * np.round(out[..., 1] / domain.height)
    return out


def _kt_geometry(
    killer_positions: np.ndarray,
    target_positions: np.ndarray,
    domain: DomainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = killer_positions[:, None, :] - target_positions[None, :, :]
    displacement = _minimum_image(displacement, domain)
    distance = np.linalg.norm(displacement, axis=2)
    return displacement, distance


def _sample_clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    if sd == 0:
        return np.full(size, np.clip(mean, lower, upper), dtype=float)
    return np.clip(rng.normal(mean, sd, size=size), lower, upper)


def _initial_positions(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Place all cells uniformly with a pair-specific non-overlap constraint."""

    n_total = config.n_killers + config.n_targets
    radii = np.concatenate(
        [
            np.full(config.n_killers, config.killer_radius),
            np.full(config.n_targets, config.target_radius),
        ]
    )
    positions = np.empty((n_total, 2), dtype=float)
    placed: list[int] = []

    for cell_index in rng.permutation(n_total):
        radius = radii[cell_index]
        if config.domain.boundary == "confined":
            x_low, x_high = config.domain.x_min + radius, config.domain.x_max - radius
            y_low, y_high = config.domain.y_min + radius, config.domain.y_max - radius
        else:
            x_low, x_high = config.domain.x_min, config.domain.x_max
            y_low, y_high = config.domain.y_min, config.domain.y_max
        if x_high <= x_low or y_high <= y_low:
            raise ValueError("the domain is too small for the configured cell radii")

        for _ in range(config.max_initialisation_attempts):
            candidate = rng.uniform([x_low, y_low], [x_high, y_high])
            if not placed:
                positions[cell_index] = candidate
                placed.append(int(cell_index))
                break
            other_indices = np.asarray(placed, dtype=int)
            displacement = candidate - positions[other_indices]
            displacement = _minimum_image(displacement, config.domain)
            distance = np.linalg.norm(displacement, axis=1)
            minimum = (
                (radius + radii[other_indices]) * config.initial_separation_factor
            )
            if np.all(distance >= minimum):
                positions[cell_index] = candidate
                placed.append(int(cell_index))
                break
        else:
            raise RuntimeError(
                "could not place all cells without overlap; enlarge the domain, "
                "reduce cell counts/radii, or lower initial_separation_factor"
            )
    return positions


def _apply_boundary(
    positions: np.ndarray,
    polarities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    domain: DomainConfig,
) -> None:
    if domain.boundary == "periodic":
        positions[alive, 0] = domain.x_min + (
            positions[alive, 0] - domain.x_min
        ) % domain.width
        positions[alive, 1] = domain.y_min + (
            positions[alive, 1] - domain.y_min
        ) % domain.height
        return

    active = np.flatnonzero(alive)
    for index in active:
        radius = radii[index]
        x_low, x_high = domain.x_min + radius, domain.x_max - radius
        y_low, y_high = domain.y_min + radius, domain.y_max - radius
        while positions[index, 0] < x_low or positions[index, 0] > x_high:
            if positions[index, 0] < x_low:
                positions[index, 0] = 2 * x_low - positions[index, 0]
            else:
                positions[index, 0] = 2 * x_high - positions[index, 0]
            polarities[index, 0] *= -1
        while positions[index, 1] < y_low or positions[index, 1] > y_high:
            if positions[index, 1] < y_low:
                positions[index, 1] = 2 * y_low - positions[index, 1]
            else:
                positions[index, 1] = 2 * y_high - positions[index, 1]
            polarities[index, 1] *= -1


def _forces(
    positions: np.ndarray,
    alive: np.ndarray,
    bound: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Return net pair forces, including periodic minimum-image geometry."""

    n_k = config.n_killers
    n_total = positions.shape[0]
    radii = np.concatenate(
        [
            np.full(n_k, config.killer_radius),
            np.full(config.n_targets, config.target_radius),
        ]
    )
    net = np.zeros_like(positions)

    for first in range(n_total - 1):
        if not alive[first]:
            continue
        for second in range(first + 1, n_total):
            if not alive[second]:
                continue
            displacement = _minimum_image(
                positions[first] - positions[second], config.domain
            )
            distance = float(np.linalg.norm(displacement))
            if distance == 0:
                continue
            unit = displacement / distance
            contact_distance = radii[first] + radii[second]

            first_is_killer = first < n_k
            second_is_killer = second < n_k
            if first_is_killer and second_is_killer:
                stiffness = config.mechanics.repulsion_kk
                is_bound = False
            elif not first_is_killer and not second_is_killer:
                stiffness = config.mechanics.repulsion_tt
                is_bound = False
            else:
                stiffness = config.mechanics.repulsion_kt
                killer_index = first if first_is_killer else second
                target_global = second if first_is_killer else first
                target_index = target_global - n_k
                is_bound = bool(bound[killer_index, target_index])

            pair_force = stiffness * max(contact_distance - distance, 0.0) * unit
            # Bound adhesion adds to repulsion, including during overlap.
            if (
                is_bound
                and distance
                < config.mechanics.capture_radius_factor * contact_distance
            ):
                pair_force -= config.mechanics.adhesion_kt * unit
            net[first] += pair_force
            net[second] -= pair_force
    return net


def _record_snapshots(
    rows: list[dict[str, float | int | str | bool]],
    *,
    step: int,
    time: float,
    positions: np.ndarray,
    polarities: np.ndarray,
    killer_state: np.ndarray,
    target_state: np.ndarray,
    target_damage: np.ndarray,
    target_alive: np.ndarray,
    config: SimulationConfig,
) -> None:
    for killer_id in range(config.n_killers):
        rows.append(
            {
                "step": step,
                "time": time,
                "cell_type": "killer",
                "cell_id": killer_id,
                "x": positions[killer_id, 0],
                "y": positions[killer_id, 1],
                "polarity_x": polarities[killer_id, 0],
                "polarity_y": polarities[killer_id, 1],
                "alive": True,
                "state": killer_state[killer_id],
                "death_factor": np.nan,
            }
        )
    for target_id in range(config.n_targets):
        global_id = config.n_killers + target_id
        rows.append(
            {
                "step": step,
                "time": time,
                "cell_type": "target",
                "cell_id": target_id,
                "x": positions[global_id, 0],
                "y": positions[global_id, 1],
                "polarity_x": polarities[global_id, 0],
                "polarity_y": polarities[global_id, 1],
                "alive": bool(target_alive[target_id]),
                "state": target_state[target_id],
                "death_factor": target_damage[target_id],
            }
        )


def _event(
    events: list[dict[str, float | int | str | None]],
    *,
    time: float,
    step: int,
    name: str,
    killer_id: int | None = None,
    target_id: int | None = None,
    value: float | None = None,
    reason: str | None = None,
) -> None:
    events.append(
        {
            "time": float(time),
            "step": int(step),
            "event": name,
            "killer_id": killer_id,
            "target_id": target_id,
            "value": value,
            "reason": reason,
        }
    )


@dataclass(slots=True)
class Simulation:
    """Executable Hookean killer-target agent-based model."""

    config: SimulationConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def run(
        self,
        *,
        initial_killer_positions: np.ndarray | None = None,
        initial_target_positions: np.ndarray | None = None,
    ) -> SimulationResult:
        config = self.config
        rng = np.random.default_rng(config.seed)
        n_k, n_t = config.n_killers, config.n_targets
        n_total = n_k + n_t

        if (initial_killer_positions is None) != (initial_target_positions is None):
            raise ValueError("provide both initial position arrays or neither")
        if initial_killer_positions is None:
            positions = _initial_positions(config, rng)
        else:
            killer_positions = np.asarray(initial_killer_positions, dtype=float)
            target_positions = np.asarray(initial_target_positions, dtype=float)
            if killer_positions.shape != (n_k, 2) or target_positions.shape != (n_t, 2):
                raise ValueError("initial position arrays have incompatible shapes")
            positions = np.vstack([killer_positions, target_positions]).copy()

        polarities = np.zeros((n_total, 2), dtype=float)
        killer_state = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_killer_state,
            config.cytotoxicity.killer_state_sd,
            n_k,
            lower=0.0,
            upper=1.0,
        )
        target_state = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_target_state,
            config.cytotoxicity.target_state_sd,
            n_t,
            lower=0.0,
            upper=1.0,
        )
        target_damage = _sample_clipped_normal(
            rng,
            config.cytotoxicity.initial_death_factor,
            config.cytotoxicity.initial_death_factor_sd,
            n_t,
            lower=0.0,
            upper=config.cytotoxicity.death_threshold,
        )
        target_alive = target_damage < config.cytotoxicity.death_threshold
        alive = np.concatenate([np.ones(n_k, dtype=bool), target_alive.copy()])
        kill_probability = _sample_clipped_normal(
            rng,
            config.cytotoxicity.kill_probability,
            config.cytotoxicity.kill_probability_sd,
            n_k,
            lower=0.0,
            upper=1.0,
        )

        bound = np.zeros((n_k, n_t), dtype=bool)
        productive = np.zeros((n_k, n_t), dtype=bool)
        damage_contribution = np.zeros((n_k, n_t), dtype=float)
        events: list[dict[str, float | int | str | None]] = []
        snapshot_rows: list[dict[str, float | int | str | bool]] = []

        _, initial_distance = _kt_geometry(positions[:n_k], positions[n_k:], config.domain)
        contact_distance = config.killer_radius + config.target_radius
        proximity_cutoff = contact_distance * (
            1.0 + config.mechanics.contact_tolerance_factor
        )
        proximity = (initial_distance <= proximity_cutoff) & target_alive[None, :]
        for killer_id, target_id in np.argwhere(proximity):
            _event(
                events,
                time=0.0,
                step=0,
                name="contact_started",
                killer_id=int(killer_id),
                target_id=int(target_id),
            )

        time = 0.0
        step = 0
        _record_snapshots(
            snapshot_rows,
            step=step,
            time=time,
            positions=positions,
            polarities=polarities,
            killer_state=killer_state,
            target_state=target_state,
            target_damage=target_damage,
            target_alive=target_alive,
            config=config,
        )

        radii = np.concatenate(
            [
                np.full(n_k, config.killer_radius),
                np.full(n_t, config.target_radius),
            ]
        )
        polarity_decay = np.concatenate(
            [
                np.full(n_k, config.motility.killer_polarity_decay),
                np.full(n_t, config.motility.target_polarity_decay),
            ]
        )[:, None]
        polarity_noise = np.concatenate(
            [
                np.full(n_k, config.motility.killer_polarity_noise),
                np.full(n_t, config.motility.target_polarity_noise),
            ]
        )[:, None]
        speed = np.concatenate(
            [
                np.full(n_k, config.motility.killer_speed),
                np.full(n_t, config.motility.target_speed),
            ]
        )[:, None]
        translation_noise = np.concatenate(
            [
                np.full(n_k, config.motility.killer_translation_noise),
                np.full(n_t, config.motility.target_translation_noise),
            ]
        )[:, None]

        while time < config.duration:
            _, distance = _kt_geometry(positions[:n_k], positions[n_k:], config.domain)
            proximity_now = (distance <= proximity_cutoff) & target_alive[None, :]
            capture_cutoff = config.mechanics.capture_radius_factor * contact_distance
            initial_forces = _forces(positions, alive, bound, config)
            deterministic_drift = speed * polarities + initial_forces
            # A bond transition can change each cell's force by F_A. Include
            # that bound before choosing dt, so probabilities use the final dt.
            switchable = (
                (
                    (bound & (config.mechanics.unbinding_rate > 0))
                    | (~bound & proximity_now & (config.mechanics.binding_rate > 0))
                )
                & (distance < capture_cutoff)
                & target_alive[None, :]
            )
            drift_bound = np.linalg.norm(deterministic_drift, axis=1)
            drift_bound[:n_k] += config.mechanics.adhesion_kt * switchable.sum(axis=1)
            drift_bound[n_k:] += config.mechanics.adhesion_kt * switchable.sum(axis=0)
            maximum_drift = float(
                np.max(drift_bound[alive], initial=0.0)
            )
            adaptive_dt = (
                config.max_drift_displacement / maximum_drift
                if maximum_drift > 0
                else config.max_dt
            )
            dt = min(config.max_dt, adaptive_dt, config.duration - time)
            next_time = time + dt
            next_step = step + 1

            p_bind = 1.0 - np.exp(-config.mechanics.binding_rate * dt)
            p_unbind = 1.0 - np.exp(-config.mechanics.unbinding_rate * dt)

            for killer_id in range(n_k):
                for target_id in range(n_t):
                    if bound[killer_id, target_id]:
                        separation = distance[killer_id, target_id] > capture_cutoff
                        stochastic = rng.random() < p_unbind
                        if not target_alive[target_id] or separation or stochastic:
                            bound[killer_id, target_id] = False
                            productive[killer_id, target_id] = False
                            _event(
                                events,
                                time=time,
                                step=step,
                                name="synapse_ended",
                                killer_id=killer_id,
                                target_id=target_id,
                                reason=(
                                    "target_death"
                                    if not target_alive[target_id]
                                    else "separation"
                                    if separation
                                    else "stochastic"
                                ),
                            )
                    elif proximity_now[killer_id, target_id] and rng.random() < p_bind:
                        bound[killer_id, target_id] = True
                        productive[killer_id, target_id] = (
                            rng.random() < kill_probability[killer_id]
                        )
                        _event(
                            events,
                            time=time,
                            step=step,
                            name="synapse_formed",
                            killer_id=killer_id,
                            target_id=target_id,
                            value=float(productive[killer_id, target_id]),
                        )
                        kill_probability[killer_id] = max(
                            0.0,
                            kill_probability[killer_id]
                            - config.cytotoxicity.kill_probability_decay_per_synapse,
                        )

            forces = _forces(positions, alive, bound, config)
            old_polarities = polarities.copy()
            polarities[alive] = (
                old_polarities[alive]
                - polarity_decay[alive] * old_polarities[alive] * dt
                + polarity_noise[alive] * np.sqrt(dt) * rng.normal(size=(alive.sum(), 2))
            )
            positions[alive] += (
                speed[alive] * old_polarities[alive] * dt
                + forces[alive] * dt
                + translation_noise[alive]
                * np.sqrt(dt)
                * rng.normal(size=(alive.sum(), 2))
            )
            _apply_boundary(positions, polarities, radii, alive, config.domain)

            exhaustion_increment = np.zeros(n_k, dtype=float)
            for target_id in range(n_t):
                if not target_alive[target_id]:
                    continue
                active_killers = np.flatnonzero(
                    bound[:, target_id] & productive[:, target_id]
                )
                individual_rates = np.array([], dtype=float)
                if active_killers.size:
                    individual_rates = target_state[target_id] * (
                        config.cytotoxicity.killing_rate_min
                        + (
                            config.cytotoxicity.killing_rate_max
                            - config.cytotoxicity.killing_rate_min
                        )
                        * killer_state[active_killers]
                    )
                    target_damage[target_id] += float(individual_rates.sum()) * dt
                    damage_contribution[active_killers, target_id] += individual_rates * dt
                    exhaustion_increment[active_killers] += (
                        config.cytotoxicity.exhaustion_rate_min
                        + (
                            config.cytotoxicity.exhaustion_rate_max
                            - config.cytotoxicity.exhaustion_rate_min
                        )
                        * target_state[target_id]
                    ) * dt
                target_damage[target_id] -= (
                    config.cytotoxicity.recovery_rate * target_damage[target_id] * dt
                )
                target_damage[target_id] = max(0.0, target_damage[target_id])

                if target_damage[target_id] >= config.cytotoxicity.death_threshold:
                    target_damage[target_id] = config.cytotoxicity.death_threshold
                    target_alive[target_id] = False
                    alive[n_k + target_id] = False
                    contributors = np.flatnonzero(damage_contribution[:, target_id] > 0)
                    primary = (
                        int(np.argmax(damage_contribution[:, target_id]))
                        if contributors.size
                        else None
                    )
                    _event(
                        events,
                        time=next_time,
                        step=next_step,
                        name="target_killed",
                        killer_id=primary,
                        target_id=target_id,
                        value=1.0,
                    )
                    _event(
                        events,
                        time=next_time,
                        step=next_step,
                        name="target_died",
                        killer_id=primary,
                        target_id=target_id,
                        value=float(target_damage[target_id]),
                    )
                    for killer_id in np.flatnonzero(bound[:, target_id]):
                        _event(
                            events,
                            time=next_time,
                            step=next_step,
                            name="synapse_ended",
                            killer_id=int(killer_id),
                            target_id=target_id,
                            reason="target_death",
                        )
                    bound[:, target_id] = False
                    productive[:, target_id] = False

            killer_state *= np.clip(1.0 - exhaustion_increment, 0.0, 1.0)
            killer_state = np.clip(killer_state, 0.0, 1.0)

            _, distance_after = _kt_geometry(
                positions[:n_k], positions[n_k:], config.domain
            )
            for killer_id, target_id in np.argwhere(
                bound & (distance_after > capture_cutoff)
            ):
                bound[killer_id, target_id] = False
                productive[killer_id, target_id] = False
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="synapse_ended",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                    reason="separation",
                )

            proximity_after = (
                distance_after <= proximity_cutoff
            ) & target_alive[None, :]
            for killer_id, target_id in np.argwhere(proximity_after & ~proximity):
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="contact_started",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                )
            for killer_id, target_id in np.argwhere(proximity & ~proximity_after):
                _event(
                    events,
                    time=next_time,
                    step=next_step,
                    name="contact_ended",
                    killer_id=int(killer_id),
                    target_id=int(target_id),
                )
            proximity = proximity_after
            time, step = next_time, next_step

            if step % config.output.record_every == 0 or time >= config.duration:
                _record_snapshots(
                    snapshot_rows,
                    step=step,
                    time=time,
                    positions=positions,
                    polarities=polarities,
                    killer_state=killer_state,
                    target_state=target_state,
                    target_damage=target_damage,
                    target_alive=target_alive,
                    config=config,
                )

        snapshots = pd.DataFrame(snapshot_rows)
        event_table = pd.DataFrame(events, columns=EVENT_COLUMNS)
        final_cells = snapshots[snapshots["time"] == snapshots["time"].max()].copy()
        final_cells.reset_index(drop=True, inplace=True)
        return SimulationResult(
            snapshots=snapshots,
            events=event_table,
            final_cells=final_cells,
            config=config,
        )


def simulate(
    config: SimulationConfig | None = None,
    **config_overrides,
) -> SimulationResult:
    """Convenience function for constructing and running a simulation."""

    if config is not None and config_overrides:
        raise ValueError("pass either config or keyword overrides, not both")
    resolved = config if config is not None else SimulationConfig(**config_overrides)
    return Simulation(resolved).run()


__all__ = ["Simulation", "simulate"]
