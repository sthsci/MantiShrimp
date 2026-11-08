"""Sweep bound killer–target adhesion in the dispersed conjugation assay."""

import argparse
from itertools import product
import json
import math
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parent / "src"
if not (SOURCE / "mantishrimp" / "__init__.py").is_file():
    raise SystemExit("Missing bundled simulator: copy the entire adhesion_sweep folder.")
sys.path.insert(0, str(SOURCE))
from mantishrimp import (
    CytotoxicityConfig, MechanicsConfig, MotilityConfig, OutputConfig, Simulation, SimulationConfig,
)
from mantishrimp.simulation import _initial_positions


def make_plan(settings, smoke=False):
    grid, seeds = settings["adhesion_kt"], settings["seeds"]
    if not isinstance(grid, list) or len(grid) < 2:
        raise ValueError("adhesion_kt must be a list with at least two values")
    for key in ("k_on", "k_off", "adhesion_kt"):
        values = grid if key == "adhesion_kt" else [settings[key]]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{key} requires finite, non-negative numbers")
    if len(set(grid)) != len(grid):
        raise ValueError("adhesion_kt must not contain duplicates")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a non-empty list of distinct non-negative integers")
    grid = sorted(grid)
    if smoke:
        grid, seeds = [grid[0], grid[-1]], seeds[:2]
    plan = pd.DataFrame(product(grid, seeds), columns=["adhesion_kt", "seed"])
    plan.insert(0, "task", range(1, len(plan) + 1))
    return plan.assign(k_on=settings["k_on"], k_off=settings["k_off"], spatial_seed=66,
                       n_killers=2 if smoke else 25, n_targets=3 if smoke else 100,
                       duration=0.4 if smoke else 100.0)


def make_config(task):
    return SimulationConfig(
        n_killers=task.n_killers, n_targets=task.n_targets, duration=task.duration,
        seed=task.seed, initial_separation_factor=2.5, max_dt=0.2,
        mechanics=MechanicsConfig(
            repulsion_kk=50, repulsion_tt=50, repulsion_kt=50,
            adhesion_kt=task.adhesion_kt, binding_rate=task.k_on, unbinding_rate=task.k_off,
        ),
        motility=MotilityConfig(
            killer_speed=50, target_speed=50, killer_polarity_decay=1, target_polarity_decay=5,
            killer_polarity_noise=0.8, target_polarity_noise=0.2,
            killer_translation_noise=1, target_translation_noise=1,
        ),
        cytotoxicity=CytotoxicityConfig(killing_rate_max=1.5, exhaustion_rate_max=0.3),
        # The endpoint and event counts need no intermediate trajectory snapshots.
        output=OutputConfig(record_every=sys.maxsize),
    )


def task_file(task_id):
    return Path("tasks") / f"{task_id:04d}.csv"


def run_task(task):
    config = make_config(task)
    sim = Simulation(config)
    positions = _initial_positions(config, np.random.default_rng(task.spatial_seed))
    print(f"Task {task.task}: adhesion_kt={task.adhesion_kt}, seed={task.seed}", flush=True)
    result = sim.run(initial_killer_positions=positions[:config.n_killers],
                     initial_target_positions=positions[config.n_killers:])
    row = {**task._asdict(), **result.summary()}
    path = task_file(task.task)
    path.parent.mkdir(exist_ok=True)
    # Only complete replicates become visible to the collection job.
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        pd.DataFrame([row]).to_csv(tmp, index=False)
    Path(tmp.name).replace(path)


def rank_results(rows):
    summary = rows.groupby(["k_on", "k_off", "duration", "adhesion_kt"], as_index=False).agg(
        n_seeds=("seed", "size"), total_killed=("targets_killed", "sum"),
        mean_fraction_killed=("fraction_targets_killed", "mean"),
        sd_fraction_killed=("fraction_targets_killed", "std"),
    )
    # Equal target and replicate counts let integer totals preserve exact ties.
    summary["best_tested"] = summary.total_killed == summary.total_killed.max()
    return summary


def report_best(summary):
    best = summary[summary.best_tested]
    values = ", ".join(f"{v:g}" for v in best.adhesion_kt)
    if len(best) == len(summary):
        interpretation = "All tested means are equal; there is no unique best tested value."
    elif summary.best_tested.iloc[[0, -1]].any():
        interpretation = "The best set includes a grid boundary; an interior optimum is not established."
    elif len(best) > 1:
        interpretation = "Multiple interior values tie. Use more seeds and a finer grid to resolve them."
    else:
        interpretation = "Interior candidate. Use more seeds and a finer grid to see whether the peak persists."
    return (f"Best tested adhesion_kt: {values}\n"
            f"Mean fraction killed: {best.mean_fraction_killed.iloc[0]:.4f} "
            f"at {best.duration.iloc[0]:g} min (n = {best.n_seeds.iloc[0]} per adhesion).\n"
            f"k_on = {best.k_on.iloc[0]:g}, k_off = {best.k_off.iloc[0]:g} min^-1.\n"
            f"{interpretation}\n")


def plot_response(rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    per_seed = rows.pivot(index="adhesion_kt", columns="seed", values="fraction_targets_killed")
    lines = ax.plot(per_seed.index, per_seed, ".-", color="0.7", linewidth=0.8)
    lines[0].set_label("Individual seeds")
    ax.plot(summary.adhesion_kt, summary.mean_fraction_killed, "o-", color="#155e75",
            label="Mean across seeds")
    best = summary[summary.best_tested]
    ax.scatter(best.adhesion_kt, best.mean_fraction_killed, s=95, facecolors="none",
               edgecolors="#a0421d", zorder=4, label="Best tested mean")
    first = rows.iloc[0]
    ax.set(xlabel="Killer–target adhesion (model units)", ylabel="Fraction of targets killed",
           ylim=(-0.025, 1.025), title=f"{first.duration:g} min   |   "
           f"$k_{{on}}$ = {first.k_on:g}, $k_{{off}}$ = {first.k_off:g} min⁻¹")
    ax.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(f"adhesion_response.{extension}", dpi=180)
    plt.close(fig)


def collect(plan):
    missing = [i for i in plan.task if not task_file(i).is_file()]
    if missing:
        raise ValueError(f"Cannot rank an incomplete sweep. Missing task IDs: {missing}")
    rows = pd.concat([pd.read_csv(task_file(i)) for i in plan.task], ignore_index=True)
    if len(rows) != len(plan) or not rows[plan.columns].eq(plan).all().all():
        raise ValueError("Replicate identities or settings do not match the saved plan.")
    if (not rows.targets_killed.between(0, rows.n_targets).all()
            or not np.allclose(rows.fraction_targets_killed, rows.targets_killed / rows.n_targets)):
        raise ValueError("A replicate has an invalid kill count or fraction.")
    summary = rank_results(rows)
    rows.to_csv("replicates.csv", index=False)
    summary.to_csv("summary.csv", index=False)
    plot_response(rows, summary)
    message = report_best(summary)
    Path("best_tested.txt").write_text(message, encoding="utf-8")
    print(message, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("--smoke-test", action="store_true")
    commands.add_parser("task").add_argument("index", type=int, help="1-based array index")
    commands.add_parser("collect")
    commands.add_parser("all")
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(json.loads(Path("settings.json").read_text()), args.smoke_test)
        plan.to_csv("plan.csv", index=False, mode="x")
        print(len(plan))
        return
    plan = pd.read_csv("plan.csv")
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
