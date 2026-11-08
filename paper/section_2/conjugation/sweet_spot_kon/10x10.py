"""Binding/unbinding sweep with the parameter_test.ipynb initial conditions.

Run every binding/unbinding combination (500 simulations with the default grids),
or use --smoke-test for one short run.
Output paths are relative to the job's working directory.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Load the simulation package shipped alongside this script.
SOURCE_DIR = Path(__file__).resolve().parent / "src"
if not (SOURCE_DIR / "mantishrimp" / "__init__.py").is_file():
    raise FileNotFoundError(
        f"MantiShrimp source package not found at {SOURCE_DIR}. "
        "Restore the src/mantishrimp directory shipped with this script."
    )
sys.path.insert(0, str(SOURCE_DIR))

from mantishrimp import (
    Simulation, SimulationConfig, DomainConfig, MechanicsConfig, MotilityConfig,
    CytotoxicityConfig, OutputConfig,
)


# Edit each axis independently; their lengths and values may differ.
K_BIND_GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
K_UNBIND_GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
SEEDS = [1, 2, 3, 4, 5]
SPATIAL_SEED = 42
TARGET_CLUSTER_RADIUS = 250.0
KILLER_RING_INNER_RADIUS = 350.0
KILLER_RING_OUTER_RADIUS = 400.0
INITIAL_KILLER_POLARITY_MAGNITUDE = 2.0
PREFIX = "paramsweep_kbind_kunbind_10x10"


def make_config(k_bind, k_unbind, seed, smoke=False):
    return SimulationConfig(
        n_killers=2 if smoke else 50,
        n_targets=3 if smoke else 200,
        killer_radius=10, target_radius=12,
        duration=0.4 if smoke else 100,
        max_dt=0.2, max_drift_displacement=1, seed=seed,
        initial_separation_factor=1,
        domain=DomainConfig(
            x_min=-410, x_max=410,
            y_min=-410, y_max=410,
            boundary="periodic",
        ),
        mechanics=MechanicsConfig(
            repulsion_kk=50, repulsion_tt=50, repulsion_kt=50,
            adhesion_kt=300, binding_rate=k_bind, unbinding_rate=k_unbind,
        ),
        motility=MotilityConfig(
            killer_speed=50, target_speed=50,
            killer_polarity_decay=1, target_polarity_decay=5.0,
            killer_polarity_noise=0.8, target_polarity_noise=0.2,
            killer_translation_noise=1, target_translation_noise=1,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_min=0,
            killing_rate_max=1.5,
            death_threshold=1,
            recovery_rate=0,
            kill_probability=1,
            kill_probability_decay_per_synapse=0,
            initial_killer_state=1,
            initial_target_state=1,
            initial_death_factor=0,
            exhaustion_rate_min=0.001, exhaustion_rate_max=0.3,
            instant_kill=False,
        ),
        output=OutputConfig(record_every=1),
    )


def make_initial_positions(config, cluster_radius, ring_inner_radius, ring_outer_radius):
    config.validate()
    region_radii = np.array([cluster_radius, ring_inner_radius, ring_outer_radius])
    if not np.all(np.isfinite(region_radii)) or not (
        0 < cluster_radius < ring_inner_radius < ring_outer_radius
    ):
        raise ValueError("Use finite radii with 0 < cluster < ring inner < ring outer.")
    domain = config.domain
    box_size = np.array([domain.width, domain.height])
    center = np.array([domain.x_min, domain.y_min]) + box_size / 2
    if max(cluster_radius + config.target_radius,
           ring_outer_radius + config.killer_radius) > box_size.min() / 2:
        raise ValueError("The initial regions, including cell radii, must fit inside the domain.")

    # Keep placement draws separate from the simulator's random stream.
    rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(1)[0])
    # Place the more crowded target disk first.
    counts = [config.n_targets, config.n_killers]
    radii = np.repeat([config.target_radius, config.killer_radius], counts)
    inner = np.repeat([0.0, ring_inner_radius], counts)
    outer = np.repeat([cluster_radius, ring_outer_radius], counts)
    positions = np.empty((sum(counts), 2))
    for i, (cell_radius, r_min, r_max) in enumerate(zip(radii, inner, outer)):
        for _ in range(config.max_initialisation_attempts):
            angle = rng.uniform(0, 2 * np.pi)
            distance = np.sqrt(rng.uniform(r_min**2, r_max**2))
            candidate = center + distance * np.array([np.cos(angle), np.sin(angle)])
            displacement = candidate - positions[:i]
            if domain.boundary == "periodic":
                displacement -= box_size * np.round(displacement / box_size)
            minimum = (cell_radius + radii[:i]) * config.initial_separation_factor
            if np.all(np.linalg.norm(displacement, axis=1) >= minimum):
                positions[i] = candidate
                break
        else:
            raise RuntimeError(
                f"Could only place {i} of {sum(counts)} cells. Enlarge the regions, "
                "reduce cell counts, or lower initial_separation_factor."
            )
    return positions[config.n_targets:].copy(), positions[:config.n_targets].copy()


def make_initial_polarities(killer_positions):
    magnitude = INITIAL_KILLER_POLARITY_MAGNITUDE
    if not np.isfinite(magnitude) or magnitude < 0:
        raise ValueError("INITIAL_KILLER_POLARITY_MAGNITUDE must be finite and non-negative.")
    toward_origin = -killer_positions
    distance_to_origin = np.linalg.norm(toward_origin, axis=1, keepdims=True)
    return magnitude * np.divide(
        toward_origin, distance_to_origin,
        out=np.zeros_like(toward_origin), where=distance_to_origin > 0,
    )


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def summarize(result, k_bind, k_unbind, seed):
    targets = result.final_cells.loc[result.final_cells["cell_type"] == "target"]
    total = len(targets)
    alive = int(targets["alive"].sum())
    kills = total - alive
    synapses = int(result.synapses_per_cell().sum())
    contacts = int(result.contacts_per_cell().sum())

    # Count bound pairs at recorded snapshots, excluding dead targets.
    # This is neither elapsed contact time nor a contact-episode count.
    snapshots = result.snapshots
    living_targets = snapshots.loc[
        (snapshots["cell_type"] == "target") & snapshots["alive"]
    ]
    legacy_samples = int(living_targets["synapses"].map(len).sum())
    row = dict(
        k_bind=k_bind, k_unbind=k_unbind, seed=seed,
        total_targets=total, alive_targets=alive, kills=kills,
        killing_efficiency=ratio(kills, total),
        contacts=legacy_samples,  # Compatibility alias; see README.
        living_bound_pair_samples=legacy_samples,
        contact_episodes=contacts, synapses=synapses,
        kills_per_synapse=ratio(kills, synapses),
        kills_per_contact=ratio(kills, legacy_samples),
        kills_per_living_bound_pair_sample=ratio(kills, legacy_samples),
        kills_per_contact_episode=ratio(kills, contacts),
        obs_time=result.duration,
    )
    killer_rows = [
        dict(k_bind=k_bind, k_unbind=k_unbind, seed=seed,
             killer_id=killer_id, kills_per_killer=int(count),
             obs_time=result.duration)
        for killer_id, count in enumerate(result.kills_per_cell())
    ]
    return row, killer_rows


def write_csv(frame, path):
    """Replace each checkpoint only after its temporary file is complete."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False,
                 compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def save_outputs(results, killers, output):
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    write_csv(pd.DataFrame(killers), output / f"{PREFIX}_killerresults.csv.gz")
    write_csv(df, output / f"{PREFIX}_HPC.csv.gz")
    summary = df.groupby(["k_bind", "k_unbind"], as_index=False).agg(
        mean_killing_efficiency=("killing_efficiency", "mean"),
        sd_killing_efficiency=("killing_efficiency", "std"),
        mean_kills=("kills", "mean"), sd_kills=("kills", "std"),
        mean_contacts=("contacts", "mean"),
        mean_living_bound_pair_samples=("living_bound_pair_samples", "mean"),
        mean_contact_episodes=("contact_episodes", "mean"),
        mean_synapses=("synapses", "mean"),
        mean_kills_per_synapse=("kills_per_synapse", "mean"),
        mean_kills_per_contact=("kills_per_contact", "mean"),
        mean_kills_per_living_bound_pair_sample=(
            "kills_per_living_bound_pair_sample", "mean"),
        mean_kills_per_contact_episode=("kills_per_contact_episode", "mean"),
        n_runs=("seed", "count"),
        obs_time=("obs_time", "first"),
    )
    write_csv(summary, output / f"{PREFIX}_summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(
        "outputs_smoke" if args.smoke_test else "outputs"
    )
    bind_grid = [0.5] if args.smoke_test else K_BIND_GRID
    unbind_grid = [0.5] if args.smoke_test else K_UNBIND_GRID
    for name, values in (("K_BIND_GRID", bind_grid), ("K_UNBIND_GRID", unbind_grid)):
        if len(values) == 0 or any(not np.isfinite(value) or value <= 0 for value in values):
            parser.error(f"{name} must contain finite positive rates")
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicate rates")
    seeds = [1] if args.smoke_test else SEEDS
    # Reproduce the notebook's seed-42 layout once, shared by all grid points.
    base = make_config(bind_grid[0], unbind_grid[0], SPATIAL_SEED, args.smoke_test)
    killer_pos, target_pos = make_initial_positions(
        base, TARGET_CLUSTER_RADIUS, KILLER_RING_INNER_RADIUS, KILLER_RING_OUTER_RADIUS,
    )
    killer_polarities = make_initial_polarities(killer_pos)
    results, killers = [], []
    total = len(bind_grid) * len(unbind_grid) * len(seeds)
    for k_bind in bind_grid:
        for k_unbind in unbind_grid:
            for seed in seeds:
                print(f"[{len(results)+1}/{total}] K_BIND={k_bind}, "
                      f"K_UNBIND={k_unbind}, seed={seed}", flush=True)
                config = make_config(k_bind, k_unbind, seed, args.smoke_test)
                result = Simulation(config).run(
                    initial_killer_positions=killer_pos,
                    initial_target_positions=target_pos,
                    initial_killer_polarities=killer_polarities,
                )
                row, per_killer = summarize(result, k_bind, k_unbind, seed)
                results.append(row)
                killers.extend(per_killer)
                save_outputs(results, killers, output)
                # Release large trajectory tables before the next simulation.
                del result
    print(f"Done. Results saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
