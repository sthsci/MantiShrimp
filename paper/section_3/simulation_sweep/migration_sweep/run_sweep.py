"""Generate reproducible killer-migration datasets for section_3 inference."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import product
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parent
SOURCE_DIR = BUNDLE / "src"
if not (SOURCE_DIR / "mantishrimp/__init__.py").is_file():
    raise FileNotFoundError(f"Copy the entire sweep folder, including {SOURCE_DIR}")
sys.path.insert(0, str(SOURCE_DIR))

import mantishrimp  # noqa: E402
from mantishrimp import (  # noqa: E402
    CytotoxicityConfig,
    DomainConfig,
    MechanicsConfig,
    MotilityConfig,
    OutputConfig,
    SimulationConfig,
    simulate,
)
from timeline import history_sink  # noqa: E402

if Path(mantishrimp.__file__).resolve() != SOURCE_DIR / "mantishrimp/__init__.py":
    raise RuntimeError("Run the sweep in a fresh Python process to use its bundled simulator.")

GRID_FILE = BUNDLE / "section2_grid.json"
GRIDS = json.loads(GRID_FILE.read_text(encoding="utf-8"))
DEFAULT_CELLS = 1000
DEFAULT_DURATION = 100.0


def notebook_config(*, n_killers=50, n_targets=200, duration=DEFAULT_DURATION, record_every=1):
    """Section 3 notebook model with the sweep's 100-time-unit exposure."""
    return SimulationConfig(
        n_killers=n_killers,
        n_targets=n_targets,
        killer_radius=12.5,
        target_radius=15.0,
        duration=duration,
        max_dt=1 / 25,
        max_drift_displacement=1.0,
        initial_separation_factor=4,
        domain=DomainConfig(
            x_min=-1200, x_max=1200, y_min=-1200, y_max=1200,
            boundary="confined",
        ),
        motility=MotilityConfig(
            killer_polarity_decay=2.5, target_polarity_decay=5.0,
            killer_polarity_noise=2.0, target_polarity_noise=0.75,
            killer_speed=175.0, target_speed=125.0,
            killer_translation_noise=0.75, target_translation_noise=0.2,
        ),
        mechanics=MechanicsConfig(binding_rate=5.0, unbinding_rate=0.5),
        cytotoxicity=CytotoxicityConfig(
            instant_kill=False,
            initial_killer_state=1.0,
            killer_state_sd=0.0,
            kill_probability=1.0,
            killing_rate_min=0.0,
            killing_rate_max=5.0,
            exhaustion_rate_min=0.01,
            exhaustion_rate_max=0.1,
            initial_target_state=1.0,
            initial_death_factor=0.0,
            death_threshold=1.0,
            recovery_rate=0.0,
        ),
        output=OutputConfig(record_every=record_every),
    )


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_settings(config, cells, seed, alphas, sigma_mus, sigma_rs):
    """Reject invalid settings before allocating output directories or running."""
    for name, value, minimum in (
        ("cells", cells, 1), ("seed", seed, 0),
        ("n_killers", config.n_killers, 1), ("n_targets", config.n_targets, 1),
        ("record_every", config.output.record_every, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, values in (
        ("alpha", alphas), ("sigma_mu", sigma_mus), ("sigma_r", sigma_rs),
    ):
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{name} must contain finite non-negative values")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicate values")
    if not math.isfinite(config.duration):
        raise ValueError("duration must be finite")
    config.validate()


RUN_FILES = ("config.json", "kills_data.csv", "final_cells.csv", "events.csv.gz",
             "snapshots.csv.gz", "steps.csv.gz", "pair_damage.csv.gz", "summary.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replicate_seed(base_seed, point, simulation):
    # A separately submitted grid slice must get the same independent layouts as the full sweep.
    key = json.dumps([base_seed, *(v + 0.0 for v in point), simulation], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big") >> 11


def run_one(job):
    index, run_id, config, directory = job
    # A scheduler kill can only leave this unpublished sibling. A retry reuses its space.
    partial = directory.with_name("." + directory.name + ".partial")
    partial.mkdir(exist_ok=True)
    write_json(partial / "config.json", config.to_dict())
    with history_sink(partial) as sink:
        result = simulate(config, step_sink=sink)
    table = pd.DataFrame({
        "simulation": run_id, "seed": config.seed, "killer_id": np.arange(config.n_killers),
        "kills": result.kills_per_cell(), "exposure": result.duration,
    })
    table.to_csv(partial / "kills_data.csv", index=False)
    result.final_cells.to_csv(partial / "final_cells.csv", index=False)
    n_steps = int(result.final_cells["step"].iloc[0])
    stride = config.output.record_every
    row = {
        "simulation": run_id, "seed": config.seed, **result.summary(),
        "n_steps": n_steps, "n_frames": 1 + (n_steps + stride - 1) // stride,
        "n_events": len(result.events), "run_path": f"runs/{directory.name}",
    }
    write_json(partial / "summary.json", row)
    (partial / "SHA256SUMS").write_text(
        "".join(f"{sha256(partial / name)}  {name}\n" for name in RUN_FILES), encoding="ascii")
    partial.rename(directory)
    return index, run_id


def verify_run(directory, run_id, config):
    checks = (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    if len(checks) != len(RUN_FILES):
        raise ValueError(f"Incomplete file inventory: {directory}")
    for name, line in zip(RUN_FILES, checks):
        if line != f"{sha256(directory / name)}  {name}":
            raise ValueError(f"Checksum mismatch: {directory / name}")
    row = read_json(directory / "summary.json")
    if (row["simulation"] != run_id or row["seed"] != config.seed
            or read_json(directory / "config.json") != config.to_dict()):
        raise ValueError(f"Run configuration mismatch: {directory}")


def pool_condition(output, condition, index, values, runs, cells):
    rows, counts = [], []
    for run_id in range(1, runs + 1):
        directory = condition / "runs" / f"run_{run_id:04d}"
        rows.append(read_json(directory / "summary.json"))
        counts.append(pd.read_csv(directory / "kills_data.csv", float_precision="round_trip"))
    pooled = pd.concat(counts, ignore_index=True).iloc[:cells]
    if len(pooled) != cells:
        raise ValueError(f"Missing killer observations: {condition}")
    write_csv(condition / "kills_data.csv", pooled)
    write_csv(condition / "runs_summary.csv", pd.DataFrame(rows))
    alpha, sigma_mu, sigma_r = values
    return {
        "condition_id": f"condition_{index + 1:04d}",
        "alpha": alpha, "sigma_mu": sigma_mu, "sigma_r": sigma_r,
        "n_observations": len(pooled), "n_simulations": runs, "exposure": rows[0]["duration"],
        "total_kills": int(pooled["kills"].sum()), "mean_kills": float(pooled["kills"].mean()),
        "zero_kill_fraction": float((pooled["kills"] == 0).mean()),
        "kills_data_path": (condition / "kills_data.csv").relative_to(output).as_posix(),
    }


def write_dataset_checksums(output, conditions, runs, manifest):
    """Reuse replicate hashes so one download check covers all tables, even missing directories."""
    temporary = output / "SHA256SUMS.tmp"
    with temporary.open("w", encoding="utf-8") as index:
        def entry(path):
            index.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")

        # The complete manifest is published last, after this inventory is safely on disk.
        index.write(f"{hashlib.sha256(json_text(manifest).encode('utf-8')).hexdigest()}  manifest.json\n")
        entry(output / "summary.csv")
        entry(output / "run_seeds.json")
        for condition in conditions:
            for name in ("config.json", "kills_data.csv", "runs_summary.csv"):
                entry(condition / name)
            for run_id in range(1, runs + 1):
                directory = condition / "runs" / f"run_{run_id:04d}"
                inventory = directory / "SHA256SUMS"
                entry(inventory)
                for line in inventory.read_text(encoding="ascii").splitlines():
                    digest, name = line.split("  ", 1)
                    index.write(f"{digest}  {(directory / name).relative_to(output).as_posix()}\n")
    temporary.replace(output / "SHA256SUMS")


def run_sweep(
    output_dir, *, alphas=GRIDS["alpha"], sigma_mus=GRIDS["sigma_mu"],
    sigma_rs=GRIDS["sigma_r"], cells=DEFAULT_CELLS, seed=42, config=None, workers=1,
    resume=False, mode="sweep",
):
    """Pool exactly cells observations; publish and recover entire simulations independently."""
    config = replace(config or notebook_config(), seed=None)
    alphas, sigma_mus, sigma_rs = map(list, (alphas, sigma_mus, sigma_rs))
    validate_settings(config, cells, seed, alphas, sigma_mus, sigma_rs)
    validate_workers(workers)
    grid = list(product(alphas, sigma_mus, sigma_rs))
    runs = (cells + config.n_killers - 1) // config.n_killers
    seeds = [replicate_seed(seed, point, run_id) for point in grid for run_id in range(1, runs + 1)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("This grid generated duplicate replicate seeds; choose another --seed")
    sources = [Path(__file__), BUNDLE / "timeline.py", GRID_FILE, *sorted((SOURCE_DIR / "mantishrimp").glob("*.py"))]
    plan = {
        "schema_version": 3, "mode": mode, "cells_per_condition": cells, "runs_per_condition": runs,
        "base_seed": seed, "alpha": alphas, "sigma_mu": sigma_mus, "sigma_r": sigma_rs,
        "base_config": config.to_dict(), "grid_order": ["alpha", "sigma_mu", "sigma_r"],
        "run_seeds_in_grid_order_file": "run_seeds.json",
        "seed_scheme": "sha256(base_seed,alpha,sigma_mu,sigma_r,simulation); first 53 bits",
        "initial_layout": "drawn independently per condition and replicate from its simulation seed",
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "source_sha256": {p.relative_to(BUNDLE).as_posix(): sha256(p) for p in sources},
    }
    output = Path(output_dir).expanduser().resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    manifest_file = output / "manifest.json"
    with (output / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another sweep is using {output}") from exc
        if resume:
            manifest = read_json(manifest_file)
            for key, value in plan.items():
                if manifest.get(key) != value:
                    raise ValueError(f"Cannot resume: {key} differs from {manifest_file}")
            if read_json(output / "run_seeds.json") != seeds:
                raise ValueError("Cannot resume: saved replicate seeds differ from this plan")
        else:
            write_json(output / "run_seeds.json", seeds)
            manifest = {**plan, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest.pop("error", None)
        previous_finish = manifest.pop("finished_at_utc", None)
        manifest.update(status="running", workers=min(workers, len(seeds)),
                        completed_runs=0, completed_conditions=0)
        write_json(manifest_file, manifest)
        conditions, jobs, summaries = [], [], {}
        completed = [0] * len(grid)

        def checkpoint(index):
            if completed[index] == runs:
                summaries[index] = pool_condition(output, conditions[index], index, grid[index], runs, cells)
                write_csv(output / "summary.csv", pd.DataFrame([summaries[i] for i in sorted(summaries)]))
            manifest.update(completed_runs=sum(completed), completed_conditions=len(summaries))
            write_json(manifest_file, manifest)

        try:
            for index, (alpha, sigma_mu, sigma_r) in enumerate(grid):
                condition = output / f"condition_{index + 1:04d}_alpha_{alpha:g}_sigma_mu_{sigma_mu:g}_sigma_r_{sigma_r:g}"
                conditions.append(condition)
                (condition / "runs").mkdir(parents=True, exist_ok=True)
                cfg = replace(config, motility=replace(config.motility, killer_polarity_decay=alpha,
                                                       killer_polarity_noise=sigma_mu, killer_translation_noise=sigma_r))
                condition_config = condition / "config.json"
                if condition_config.exists():
                    if read_json(condition_config) != cfg.to_dict():
                        raise ValueError(f"Condition configuration mismatch: {condition_config}")
                else:
                    write_json(condition_config, cfg.to_dict())
                for run_id in range(1, runs + 1):
                    run_config = replace(cfg, seed=seeds[index * runs + run_id - 1])
                    directory = condition / "runs" / f"run_{run_id:04d}"
                    if directory.exists():
                        verify_run(directory, run_id, run_config)
                        completed[index] += 1
                    else:
                        jobs.append((index, run_id, run_config, directory))
                if completed[index]:
                    checkpoint(index)

            print(f"{output}: {sum(completed)} verified, {len(jobs)} simulations to run.", flush=True)
            active = min(workers, len(jobs))
            with (ProcessPoolExecutor(active, mp_context=get_context("spawn")) if active > 1 else nullcontext()) as pool:
                futures = [pool.submit(run_one, job) for job in jobs] if pool else []
                outcomes = (f.result() for f in as_completed(futures)) if pool else map(run_one, jobs)
                try:
                    for index, run_id in outcomes:
                        completed[index] += 1
                        checkpoint(index)
                        print(f"[{manifest['completed_runs']}/{len(seeds)}] saved condition {index + 1}, "
                              f"simulation {run_id}", flush=True)
                finally:
                    for future in futures:
                        future.cancel()
            finish = previous_finish if not jobs and previous_finish else datetime.now(timezone.utc).isoformat()
            manifest.update(status="complete", finished_at_utc=finish)
            write_dataset_checksums(output, conditions, runs, manifest)
        except BaseException as exc:
            manifest.pop("finished_at_utc", None)
            manifest.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(manifest_file, manifest)
    print(f"Complete: {output / 'summary.csv'}", flush=True)
    return output


def validate_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    keys = ["LSB_DJOB_NUMPROC", "PBS_NCPUS"]
    if os.getenv("PBS_JOBID"):
        keys.append("NCPUS")
    for key in keys:
        slots = os.getenv(key, "")
        if slots.isdecimal() and 0 < int(slots) < workers:
            raise ValueError(f"--workers {workers} exceeds the {slots} CPUs reported by {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cells", type=int, help=f"Pooled killer observations per condition (default: {DEFAULT_CELLS}).")
    parser.add_argument("--alpha", nargs="+", type=float)
    parser.add_argument("--sigma-mu", nargs="+", type=float)
    parser.add_argument("--sigma-r", nargs="+", type=float, help="Killer translational noise grid.")
    parser.add_argument("--killers-per-run", type=int)
    parser.add_argument("--targets-per-run", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--record-every", type=int, help="Cell snapshot stride in steps (default: 1).")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent simulations on one node.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke-test", action="store_true", help="Two short reduced-population simulations.")
    mode_group.add_argument("--pilot", action="store_true", help="One simulation at each grid extreme combination.")
    dest = parser.add_mutually_exclusive_group()
    dest.add_argument("--output-dir", type=Path, help="New dataset directory.")
    dest.add_argument("--resume", type=Path, metavar="DATA", help="Verify completed runs and finish this dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without creating files.")
    args = parser.parse_args()
    settings = {k: v for k, v in vars(args).items() if v is not None and k in {
        "cells", "alpha", "sigma_mu", "sigma_r", "killers_per_run", "targets_per_run",
        "duration", "record_every", "seed"}}
    try:
        validate_workers(args.workers)
        if args.resume:
            if settings or args.smoke_test or args.pilot:
                parser.error("--resume uses the saved settings; only --workers and --dry-run may change")
            m = read_json(args.resume / "manifest.json")
            if m.get("schema_version") != 3:
                parser.error("use the run's archived source to resume an older dataset")
            base = m["base_config"].copy()
            for key, cls in (("domain", DomainConfig), ("motility", MotilityConfig), ("mechanics", MechanicsConfig),
                             ("cytotoxicity", CytotoxicityConfig), ("output", OutputConfig)):
                base[key] = cls(**base[key])
            config = SimulationConfig(**base)
            alphas, mus, rs = m["alpha"], m["sigma_mu"], m["sigma_r"]
            cells, seed, mode = m["cells_per_condition"], m["base_seed"], m["mode"]
        else:
            if args.smoke_test and settings.keys() - {"seed"}:
                parser.error("--smoke-test is a fixed preset; only --seed may override its settings")
            if args.pilot and args.cells is not None:
                parser.error("--pilot uses one full population per condition; omit --cells")
            get = settings.get
            alphas = get("alpha", GRIDS["alpha"])
            mus = get("sigma_mu", GRIDS["sigma_mu"])
            rs = get("sigma_r", GRIDS["sigma_r"])
            cells, seed, mode = get("cells", DEFAULT_CELLS), get("seed", 42), "sweep"
            config = notebook_config(n_killers=get("killers_per_run", 50), n_targets=get("targets_per_run", 200),
                                     duration=get("duration", DEFAULT_DURATION), record_every=get("record_every", 1))
            if args.smoke_test:
                config = notebook_config(n_killers=2, n_targets=3, duration=0.08)
                alphas, mus, rs, cells, mode = [1.0], [0.8], [1.0], 3, "smoke"
        validate_settings(config, cells, seed, alphas, mus, rs)
        if args.pilot:
            alphas, mus, rs = [sorted({min(axis), max(axis)}) for axis in (alphas, mus, rs)]
            cells, mode = config.n_killers, "pilot"
        n = len(alphas) * len(mus) * len(rs)
        runs = (cells + config.n_killers - 1) // config.n_killers
        print(f"{mode}: {n} conditions x {runs} simulations = {n * runs} simulations; "
              f"{cells} observations/condition, {min(args.workers, n * runs)} worker(s).\n"
              f"Each simulation: {config.n_killers} killers + {config.n_targets} targets, "
              f"duration={config.duration:g}, cell snapshot every {config.output.record_every} step(s).\n"
              f"alpha={alphas}; sigma_mu={mus}; sigma_r={rs}; fixed killer speed={config.motility.killer_speed:g}.\n"
              f"Fresh layout per replicate and condition; base seed={seed}.", flush=True)
        if args.dry_run:
            return
        output = args.resume or args.output_dir or (
            BUNDLE / "outputs" / f"sweep_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}")
        run_sweep(output, alphas=alphas, sigma_mus=mus, sigma_rs=rs, cells=cells, seed=seed,
                  config=config, workers=args.workers, resume=args.resume is not None, mode=mode)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Sweep error: {exc}\n")


if __name__ == "__main__":
    main()
