"""Binding/unbinding sweep against this checkout's src/mantishrimp package.

Run normally for 500 simulations, or use --smoke-test for one short run.
Output paths are relative to the job's working directory.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# HPC layout: fyp/sweep_kbind_kunbind.py alongside fyp/MantiShrimp/.
# Use the explicit source path independently of the job's working directory.
SOURCE_DIR = Path("/rds/general/user/ss11123/home/fyp/MantiShrimp/src")
if not (SOURCE_DIR / "mantishrimp" / "__init__.py").is_file():
    raise FileNotFoundError(
        f"MantiShrimp source package not found at {SOURCE_DIR}. "
        "Upload the current repository there or correct SOURCE_DIR in this script."
    )
sys.path.insert(0, str(SOURCE_DIR))

from mantishrimp import (
    Simulation, SimulationConfig, MechanicsConfig, MotilityConfig,
    CytotoxicityConfig, OutputConfig,
)
# Private helper from the specified source tree; preserves its placement rules.
from mantishrimp.simulation import _initial_positions


GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10]
SPATIAL_SEED = 42
PREFIX = "paramsweep_kbind_kunbind_10x10"


def make_config(k_bind, k_unbind, seed, smoke=False):
    return SimulationConfig(
        n_killers=2 if smoke else 25,
        n_targets=3 if smoke else 100,
        killer_radius=10, target_radius=12,
        duration=0.4 if smoke else 25,
        max_dt=0.2, max_drift_displacement=1, seed=seed,
        initial_separation_factor=2.5,
        mechanics=MechanicsConfig(
            repulsion_kk=250, repulsion_tt=250, repulsion_kt=250,
            adhesion_kt=25, binding_rate=k_bind, unbinding_rate=k_unbind,
        ),
        motility=MotilityConfig(
            killer_speed=150, target_speed=50,
            killer_polarity_decay=2.5, target_polarity_decay=2.5,
            killer_polarity_noise=1, target_polarity_noise=1,
            killer_translation_noise=1, target_translation_noise=1,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_min=0, killing_rate_max=1, death_threshold=1,
            recovery_rate=0, kill_probability=1,
            kill_probability_decay_per_synapse=0.05,
            initial_killer_state=1, initial_target_state=1,
            exhaustion_rate_min=0.001, exhaustion_rate_max=0.1,
            instant_kill=False,
        ),
        output=OutputConfig(record_every=1),
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
    grid = [0.5] if args.smoke_test else GRID
    seeds = [1] if args.smoke_test else [1, 2, 3, 4, 5]
    base = make_config(grid[0], grid[0], seeds[0], args.smoke_test)
    base.validate()
    positions = _initial_positions(base, np.random.default_rng(SPATIAL_SEED))
    killer_pos = positions[:base.n_killers]
    target_pos = positions[base.n_killers:]
    results, killers = [], []
    total = len(grid) ** 2 * len(seeds)
    for k_bind in grid:
        for k_unbind in grid:
            for seed in seeds:
                print(f"[{len(results)+1}/{total}] K_BIND={k_bind}, "
                      f"K_UNBIND={k_unbind}, seed={seed}", flush=True)
                config = make_config(k_bind, k_unbind, seed, args.smoke_test)
                result = Simulation(config).run(
                    initial_killer_positions=killer_pos,
                    initial_target_positions=target_pos,
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
