"""Run the Cartesian sweep of unbinding rate and killer–target adhesion."""

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from assay import simulate_task

ROOT = Path(__file__).resolve().parent
AXES = ["k_off", "adhesion_kt"]
CONDITION = ["k_on", "sigma_mu", "spatial_seed", "n_killers", "n_targets", "duration"]
STATS = ["n_seeds", "total_killed", "mean_fraction_killed", "sd_fraction_killed"]
REFERENCE = "adhesion_sweep/full_20261105T120153Z_CwBwrf"


def make_plan(settings, smoke=False):
    axes = []
    for key in AXES:
        values = settings[key]
        if (not isinstance(values, list) or len(values) < 2 or
                any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{key}: supply at least two distinct finite non-negative values")
        values = sorted(values)
        axes.append([values[0], values[-1]] if smoke else values)
    seeds = settings["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct non-negative integers")
    k_on, layout = settings["k_on"], settings["spatial_seed"]
    sigma_mu = settings.get("sigma_mu", 0.8)  # Earlier settings used a fixed killer noise of 0.8.
    for name, value in (("k_on", k_on), ("sigma_mu", sigma_mu)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if type(layout) is not int or layout < 0 or type(settings["reuse_reference"]) is not bool:
        raise ValueError("spatial_seed must be a non-negative integer; reuse_reference must be boolean")
    conditions = dict(k_on=k_on, sigma_mu=sigma_mu, spatial_seed=layout, n_killers=2 if smoke else 25,
                      n_targets=3 if smoke else 100, duration=0.4 if smoke else 100.0)
    plan = pd.DataFrame(product(*axes, seeds[:2] if smoke else seeds), columns=AXES + ["seed"])
    plan = plan.assign(**conditions)
    reused = pd.DataFrame(columns=AXES + CONDITION + STATS + ["source"])
    if (not smoke and settings["reuse_reference"] and layout == 66 and k_on == 5 and sigma_mu == 0.8
            and sorted(seeds) == [1, 2, 3, 4, 5] and 0.01 in axes[0]):
        # An edited assay or baseline must not silently contribute historical measurements.
        for line in (ROOT / "reference/SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{name} differs from the baseline; set reuse_reference to false")
        old = pd.read_csv(ROOT / "reference/adhesion_summary.csv")
        old = old.loc[old.adhesion_kt.isin(axes[1])]
        reused = old[AXES + STATS].assign(**conditions, source=REFERENCE)[reused.columns]
        plan = plan[~((plan.k_off == 0.01) & plan.adhesion_kt.isin(reused.adhesion_kt))]
    plan = plan.reset_index(drop=True)
    plan.insert(0, "task", range(1, len(plan) + 1))
    return plan, reused


def task_file(task_id):
    return Path("tasks") / f"{task_id:04d}.csv"


def run_task(task):
    print(f"Task {task.task}: k_off={task.k_off:g}, adhesion={task.adhesion_kt:g}, "
          f"seed={task.seed}", flush=True)
    row = {**task._asdict(), **simulate_task(task)}
    path = task_file(task.task)
    path.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        pd.DataFrame([row]).to_csv(tmp, index=False)
    Path(tmp.name).replace(path)


def collect(plan):
    expected, reused = make_plan(json.loads(Path("settings.json").read_text()),
                                 smoke=plan.duration.iat[0] == 0.4)
    if plan.shape != expected.shape or not plan.eq(expected).all().all():
        raise ValueError("Task plan does not match the saved settings")
    missing = [i for i in plan.task if not task_file(i).is_file()]
    if missing:
        raise ValueError(f"Incomplete sweep; missing task IDs: {missing}")
    rows = pd.concat([pd.read_csv(task_file(i), float_precision="round_trip") for i in plan.task],
                     ignore_index=True)
    if len(rows) != len(plan) or not rows[plan.columns].eq(plan).all().all():
        raise ValueError("Replicate identities or settings do not match plan.csv")
    if (not rows.targets_killed.between(0, rows.n_targets).all() or (rows.targets_killed % 1).any()
            or not np.allclose(rows.fraction_targets_killed, rows.targets_killed / rows.n_targets)):
        raise ValueError("Invalid target count or kill fraction")
    summary = rows.groupby(AXES + CONDITION, as_index=False).agg(
        n_seeds=("seed", "size"), total_killed=("targets_killed", "sum"),
        mean_fraction_killed=("fraction_targets_killed", "mean"),
        sd_fraction_killed=("fraction_targets_killed", "std")).assign(source="new")
    if not reused.empty:
        summary = pd.concat([summary, reused], ignore_index=True)
    summary = summary.sort_values(AXES)
    rows.to_csv("replicates.csv", index=False)
    summary.to_csv("summary.csv", index=False)
    from plot_sweep import plot
    plot(summary, Path("."))
    print(f"{len(summary)} grid cells; {len(rows)} new replicates. Results: {Path.cwd()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("--smoke-test", action="store_true")
    commands.add_parser("task").add_argument("index", type=int)
    commands.add_parser("collect")
    commands.add_parser("all")
    args = parser.parse_args()
    if args.command == "plan":
        plan, reused = make_plan(json.loads(Path("settings.json").read_text()), args.smoke_test)
        plan.to_csv("plan.csv", index=False, mode="x")
        reused.to_csv("reference_used.csv", index=False, mode="x")
        print(len(plan))
        return
    plan = pd.read_csv("plan.csv", float_precision="round_trip")
    if args.command == "task":
        if not 1 <= args.index <= len(plan):
            parser.error(f"task index must be between 1 and {len(plan)}")
        run_task(next(plan.iloc[args.index - 1:args.index].itertuples(index=False)))
    else:
        if args.command == "all":
            for task in plan.itertuples(index=False):
                run_task(task)
        collect(plan)


if __name__ == "__main__":
    main()
