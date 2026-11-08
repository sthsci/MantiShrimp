"""Sweep killer polarity decay, polarity noise and translational noise.

Run with --smoke-test for one short execution check. Edit the three grids and
SEEDS below for a full Cartesian sweep. All other settings follow the supplied
binding/unbinding sweep, with binding and unbinding fixed at 5 and 0.5.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[3] / "src"
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
    raise RuntimeError("Restart Python to load this checkout's simulator.")


ALPHA_GRID = [0.5, 1, 1.5, 2, 2.5, 3]
SIGMA_MU_GRID = [0.1, 0.25, 0.5, 0.75, 0.8, 1, 1.5]
SIGMA_R_GRID = [0, 0.2, 1, 5, 10, 50]
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
        output=OutputConfig(record_every=1),
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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are refused.")
    args = parser.parse_args()
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
    base = make_config(1, 0.8, 1, seeds[0], args.smoke_test)
    base.validate()
    output = args.output_dir or Path(__file__).resolve().parent / (
        "outputs_motility_smoke" if args.smoke_test else "outputs_motility"
    )
    if output.exists():
        parser.error(f"{output} already exists; choose a new --output-dir")
    positions = _initial_positions(base, np.random.default_rng(SPATIAL_SEED))
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({
        "grids": grids, "seeds": seeds, "spatial_seed": SPATIAL_SEED,
        "smoke_test": args.smoke_test, "base_config": base.to_dict(),
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    results, killers = [], []
    total = np.prod([len(values) for values in grids.values()]) * len(seeds)
    for alpha, sigma_mu, sigma_r, seed in product(*grids.values(), seeds):
        print(f"[{len(results) + 1}/{total}] alpha={alpha:g}, "
              f"sigma_mu={sigma_mu:g}, sigma_r={sigma_r:g}, seed={seed}", flush=True)
        config = make_config(alpha, sigma_mu, sigma_r, seed, args.smoke_test)
        result = Simulation(config).run(
            initial_killer_positions=positions[:base.n_killers],
            initial_target_positions=positions[base.n_killers:],
        )
        row, per_killer = summarize(result)
        results.append(row)
        killers.extend(per_killer)
        save_outputs(results, killers, output)
        del result
    print(f"Done. Results saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
