"""Sweep killer polarity decay, polarity noise and translational noise.

Run with --smoke-test for one short execution check, or --pilot for 16
full-size simulations at the grid extremes. The full sweep runs every
combination of ALPHA_GRID, SIGMA_MU_GRID, SIGMA_R_GRID and SEEDS.
Use --workers to run independent simulations in parallel on one compute node.
Other settings follow the binding/unbinding sweep, with rates fixed at 5 and 0.5.
"""

import argparse
from contextlib import nullcontext
from itertools import product
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent / "src"
if not (SOURCE_DIR / "mantishrimp/__init__.py").is_file():
    raise FileNotFoundError(f"MantiShrimp source package not found at {SOURCE_DIR}.")
sys.path.insert(0, str(SOURCE_DIR))

import mantishrimp
from mantishrimp import (
    CytotoxicityConfig, DomainConfig, MechanicsConfig, MotilityConfig,
    OutputConfig, Simulation, SimulationConfig,
)
from mantishrimp.simulation import _initial_positions

if Path(mantishrimp.__file__).resolve() != SOURCE_DIR / "mantishrimp/__init__.py":
    raise RuntimeError("Restart Python to load the simulator bundled with this sweep.")


# Include both reference heatmap slices: sigma_mu = 0.8 and sigma_r = 1.
ALPHA_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00]
SIGMA_MU_GRID = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5]
SIGMA_R_GRID = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50]
SEEDS = [1, 2, 3, 4, 5]
SPATIAL_SEED = 42
PARAMETERS = ["alpha", "sigma_mu", "sigma_r"]


def make_config(alpha, sigma_mu, sigma_r, seed, smoke=False):
    return SimulationConfig(
        n_killers=2 if smoke else 25,
        n_targets=3 if smoke else 100,
        killer_radius=10, target_radius=12,
        duration=0.4 if smoke else 100,
        max_dt=0.2, max_drift_displacement=1, seed=seed,
        initial_separation_factor=2.5,
        domain=DomainConfig(
            x_min=-600, x_max=600, y_min=-600, y_max=600,
            boundary="periodic",
        ),
        mechanics=MechanicsConfig(
            repulsion_kk=50, repulsion_tt=50, repulsion_kt=50,
            adhesion_kt=300, binding_rate=5, unbinding_rate=0.5,
        ),
        motility=MotilityConfig(
            killer_speed=50, target_speed=50,
            killer_polarity_decay=alpha, target_polarity_decay=5,
            killer_polarity_noise=sigma_mu, target_polarity_noise=0.2,
            killer_translation_noise=sigma_r, target_translation_noise=1,
        ),
        cytotoxicity=CytotoxicityConfig(
            killing_rate_min=0, killing_rate_max=1.5,
            death_threshold=1, recovery_rate=0, kill_probability=1,
            kill_probability_decay_per_synapse=0,
            initial_killer_state=1, initial_target_state=1, initial_death_factor=0,
            exhaustion_rate_min=0.001, exhaustion_rate_max=0.3,
            instant_kill=False,
        ),
        # Event counts do not depend on trajectory recording frequency.
        output=OutputConfig(record_every=100),
    )


def summarize(result):
    config = result.config
    parameters = dict(
        alpha=config.motility.killer_polarity_decay,
        sigma_mu=config.motility.killer_polarity_noise,
        sigma_r=config.motility.killer_translation_noise,
        seed=config.seed,
    )
    contact_events = result.events.loc[result.events["event"].eq("contact_started")]
    unique = (
        contact_events.groupby("killer_id")["target_id"].nunique()
        .reindex(range(config.n_killers), fill_value=0).to_numpy(dtype=int)
    )
    per_killer = pd.DataFrame({
        **parameters,
        "killer_id": np.arange(config.n_killers),
        "unique_contacts": unique,
        "contact_episodes": result.contacts_per_cell(),
        "synapses": result.synapses_per_cell(),
        "kills": result.kills_per_cell(),
        "obs_time": result.duration,
    })
    targets = result.final_cells.loc[result.final_cells["cell_type"].eq("target")]
    alive = int(targets["alive"].sum())
    kills = len(targets) - alive
    row = dict(
        **parameters,
        n_killers=config.n_killers, total_targets=len(targets), alive_targets=alive,
        kills=kills, killing_efficiency=kills / len(targets),
        unique_contacts=int(unique.sum()),
        unique_contacts_per_killer=float(unique.mean()),
        contact_episodes=int(per_killer["contact_episodes"].sum()),
        synapses=int(per_killer["synapses"].sum()),
        kills_per_killer=float(per_killer["kills"].mean()),
        obs_time=result.duration,
    )
    return row, per_killer.to_dict("records")


def run_one(job):
    """Save this simulation's events before returning its counts to the parent."""
    number, config, positions, output = job
    result = Simulation(config).run(
        initial_killer_positions=positions[:config.n_killers],
        initial_target_positions=positions[config.n_killers:],
    )
    row, per_killer = summarize(result)
    events_file = f"events/{number:06d}.csv.gz"
    events = result.events.assign(**{key: row[key] for key in (*PARAMETERS, "seed")})
    write_csv(events, output / events_file)
    row.update(events_file=events_file, n_events=len(events))
    return row, per_killer


def write_csv(frame, path):
    """Replace a checkpoint only after its temporary file is complete."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False,
                 compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def save_outputs(results, killers, output):
    runs = pd.DataFrame(results)
    metrics = (
        "unique_contacts_per_killer", "killing_efficiency", "kills_per_killer",
        "kills", "contact_episodes", "synapses",
    )
    summary = runs.groupby(PARAMETERS, as_index=False).agg(
        **{f"{label}_{metric}": (metric, statistic)
           for metric in metrics
           for label, statistic in (("mean", "mean"), ("sd", "std"))},
        n_runs=("seed", "count"),
    )
    write_csv(pd.DataFrame(killers), output / "killers.csv.gz")
    write_csv(summary, output / "summary.csv")
    write_csv(runs, output / "runs.csv.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true",
                      help="Full-size runs at each grid's min/max with the first two seeds.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel simulations; must not exceed allocated CPU cores.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate settings and print the plan without running or saving.")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are refused.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    grids = dict(zip(PARAMETERS, (
        ([1], [0.8], [1]) if args.smoke_test
        else (ALPHA_GRID, SIGMA_MU_GRID, SIGMA_R_GRID)
    )))
    seeds = [1] if args.smoke_test else SEEDS
    for name, values in grids.items():
        if not values or any(not np.isfinite(v) or v < 0 for v in values):
            parser.error(f"{name} must contain finite non-negative values")
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int)
                         or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        parser.error("SEEDS must contain distinct non-negative integers")
    if args.pilot:
        if any(len(values) < 2 for values in grids.values()) or len(seeds) < 2:
            parser.error("--pilot needs at least two values per grid and two seeds")
        grids = {name: [min(values), max(values)] for name, values in grids.items()}
        seeds = seeds[:2]
    base = make_config(1, 0.8, 1, seeds[0], args.smoke_test)
    base.validate()
    total = int(np.prod([len(values) for values in grids.values()])) * len(seeds)
    workers = min(args.workers, total)
    print(f"Plan: {total} simulations, {workers} worker(s), "
          f"{base.n_killers} killers, {base.n_targets} targets, duration={base.duration:g}.",
          flush=True)
    if args.pilot:
        print(f"PILOT: grids={grids}; seeds={seeds}", flush=True)
    if args.dry_run:
        return
    output = args.output_dir or Path(__file__).resolve().parent / (
        "outputs_motility_smoke" if args.smoke_test else
        "outputs_motility_pilot" if args.pilot else "outputs_motility"
    )
    if output.exists():
        parser.error(f"{output} already exists; choose a new --output-dir")
    positions = _initial_positions(base, np.random.default_rng(SPATIAL_SEED))
    output.mkdir(parents=True, exist_ok=False)
    (output / "events").mkdir()
    (output / "settings.json").write_text(json.dumps({
        "grids": grids, "seeds": seeds, "spatial_seed": SPATIAL_SEED,
        "smoke_test": args.smoke_test, "pilot": args.pilot,
        "workers": workers, "base_config": base.to_dict(),
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    results, killers = [], []
    jobs = ((number, make_config(alpha, mu, r, seed, args.smoke_test), positions, output)
            for number, (alpha, mu, r, seed) in enumerate(product(*grids.values(), seeds), 1))
    # Ordered results keep serial and parallel checkpoints reproducible.
    with (nullcontext() if workers == 1 else get_context("spawn").Pool(workers)) as pool:
        outcomes = map(run_one, jobs) if pool is None else pool.imap(run_one, jobs, chunksize=1)
        for row, per_killer in outcomes:
            results.append(row)
            killers.extend(per_killer)
            save_outputs(results, killers, output)
            print(f"[{len(results)}/{total}] saved alpha={row['alpha']:g}, "
                  f"sigma_mu={row['sigma_mu']:g}, sigma_r={row['sigma_r']:g}, "
                  f"seed={row['seed']}", flush=True)
    print(f"Done. Results saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
