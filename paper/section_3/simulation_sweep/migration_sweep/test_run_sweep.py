"""Small sweep checks; no Bayesian sampling or full-size simulations."""

import ast
from dataclasses import replace
import gzip
import io
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import run_sweep as sweep
from mantishrimp import Simulation


@pytest.fixture
def tracked_simulator(monkeypatch):
    calls = []
    original = sweep.simulate

    def simulate(config, **kwargs):
        calls.append(config)
        return original(config, **kwargs)

    monkeypatch.setattr(sweep, "simulate", simulate)
    return calls


def test_baseline_matches_notebook():
    notebook = sweep.BUNDLE.parent.parent / "synthetic_data_inference.ipynb"
    if not notebook.is_file():
        pytest.skip("Optional cross-check against the notebook in the complete checkout")
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "config = SimulationConfig(" in "".join(c["source"]))
    assignment = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets)
    )
    # Evaluate only the configuration expression, without running notebook imports or simulations.
    reference = eval(compile(ast.Expression(assignment.value), str(notebook), "eval"), vars(sweep))
    actual = sweep.notebook_config()
    assert actual.duration == 100
    assert replace(actual, seed=reference.seed, duration=reference.duration) == reference


def test_grid_matches_section2_production_run():
    settings = (sweep.BUNDLE.parents[2] / "section_2/migration/mig_sweep/runs"
                / "run_5000_sims_10x10x10/data/settings.json")
    if not settings.is_file():
        pytest.skip("Optional cross-check against the archived Section 2 sweep")
    assert sweep.GRIDS == sweep.read_json(settings)["grids"]
    assert list(sweep.GRIDS) == ["alpha", "sigma_mu", "sigma_r"]
    assert all(len(set(axis)) == 10 for axis in sweep.GRIDS.values())


def test_shell_submission_routes_each_job_to_ephemeral(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(sweep.BUNDLE, bundle, ignore=shutil.ignore_patterns(".*", "__pycache__", "outputs", "runs"))
    venv = bundle / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    scratch = tmp_path / "ephemeral"
    scratch.mkdir()
    # Stand in for the login shell's python ~/bin/bsub; never contact a scheduler.
    wrapper = tmp_path / "python"
    wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$EPHEMERAL/calls"\n'
                       'test -z "$REJECT" && echo "1234.pbs-7" || echo "qsub rejected the job"\n')
    wrapper.chmod(0o755)
    (tmp_path / "qsub").symlink_to(wrapper)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", EPHEMERAL=str(scratch),
               SWEEP_PYTHON=sys.executable, REJECT="")
    cmd = ["bash", str(bundle / "submit_full_sweep.sh")]
    subprocess.run(cmd, env=env, check=True, capture_output=True)
    batch = scratch / "section3_grid10"
    submitted = [line.split() for line in (batch / "submitted.tsv").read_text().splitlines()]
    assert [(float(a), float(m)) for _, a, m, _ in submitted] == list(product(sweep.GRIDS["alpha"], sweep.GRIDS["sigma_mu"]))
    calls = (scratch / "calls").read_text().splitlines()
    assert [v for v in calls if v.startswith("SWEEP_RUNS_DIR=")] == [f"SWEEP_RUNS_DIR={batch}/{i:03d}" for i in range(1, 101)]
    assert subprocess.run(cmd, env=env, capture_output=True).returncode
    assert (scratch / "calls").read_text().splitlines() == calls
    scratch2 = tmp_path / "rejected"
    scratch2.mkdir()
    failure = subprocess.run(cmd, env={**env, "EPHEMERAL": str(scratch2), "REJECT": "1"}, capture_output=True, text=True)
    assert failure.returncode and "No PBS job ID for slice 001" in failure.stderr
    assert not (scratch2 / "section3_grid10/submitted.tsv").exists()


def test_grid_pooling_provenance_and_repeatability(tmp_path, tracked_simulator):
    config = sweep.notebook_config(n_killers=3, n_targets=12, duration=0.05)
    kwargs = dict(config=config, cells=5, seed=42, alphas=[1.25, 2.5], sigma_mus=[0.2, 0.8], sigma_rs=[0, 15])
    first = sweep.run_sweep(tmp_path / "first", **kwargs)
    second = sweep.run_sweep(tmp_path / "second", **kwargs)
    summary = pd.read_csv(first / "summary.csv")
    assert list(zip(summary.alpha, summary.sigma_mu, summary.sigma_r)) == list(product(kwargs["alphas"], kwargs["sigma_mus"], kwargs["sigma_rs"]))
    assert len(tracked_simulator) == 32
    layouts = set()
    for index, row in summary.iterrows():
        condition = (first / row.kills_data_path).parent
        pooled = pd.read_csv(condition / "kills_data.csv", dtype={"seed": str})
        assert list(pooled.columns) == ["simulation", "seed", "killer_id", "kills", "exposure"]
        assert len(pooled) == 5
        assert pooled.simulation.tolist() == [1, 1, 1, 2, 2]
        assert pooled.killer_id.tolist() == [0, 1, 2, 0, 1]
        assert (pooled.loc[pooled.killer_id == 0, "kills"] == 0).all()
        assert (pooled.exposure == config.duration).all()
        for run_id in (1, 2):
            run = condition / "runs" / f"run_{run_id:04d}"
            raw = pd.read_csv(run / "kills_data.csv", dtype={"seed": str})
            saved = json.loads((run / "config.json").read_text())
            expected_seed = str(tracked_simulator[index * 2 + run_id - 1].seed)
            assert len(raw) == config.n_killers
            assert raw.seed.tolist() == [expected_seed] * config.n_killers
            assert set(pooled.loc[pooled.simulation == run_id, "seed"]) == {expected_seed}
            assert str(saved["seed"]) == expected_seed
            assert saved["motility"] == {
                **config.to_dict()["motility"], "killer_polarity_decay": row.alpha,
                "killer_polarity_noise": row.sigma_mu, "killer_translation_noise": row.sigma_r,
            }
            initial = pd.read_csv(run / "snapshots.csv.gz", nrows=config.n_killers + config.n_targets)
            assert initial.step.eq(0).all()
            layouts.add(initial[["x", "y"]].to_numpy().tobytes())
            assert (run / "summary.json").is_file()
        assert (first / row.kills_data_path).read_bytes() == (second / row.kills_data_path).read_bytes()
    assert config.motility == sweep.notebook_config().motility
    manifests = [json.loads((path / "manifest.json").read_text()) for path in (first, second)]
    seed_lists = [sweep.read_json(path / manifest["run_seeds_in_grid_order_file"])
                  for path, manifest in zip((first, second), manifests)]
    assert seed_lists[0] == seed_lists[1]
    assert len(layouts) == len(set(seed_lists[0])) == 16
    assert manifests[0]["status"] == "complete"
    assert manifests[0]["completed_conditions"] == 8

    # Submitting one slice, even with a larger sample target, keeps overlapping replicates identical.
    sliced = sweep.run_sweep(tmp_path / "slice", **{**kwargs, "alphas": [2.5], "sigma_mus": [0.8],
                                                    "sigma_rs": [15], "cells": 8})
    selected = (first / summary.iloc[-1].kills_data_path).parent
    sliced_condition, = sliced.glob("condition_*")
    for run in (selected / "runs").iterdir():
        assert (run / "SHA256SUMS").read_bytes() == (sliced_condition / "runs" / run.name / "SHA256SUMS").read_bytes()


def test_existing_output_is_preserved(tmp_path, tracked_simulator):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        sweep.run_sweep(existing, cells=1)
    assert sentinel.read_text() == "original"
    assert list(existing.iterdir()) == [sentinel]
    assert not tracked_simulator


@pytest.mark.parametrize("kwargs", [
    {"cells": 0}, {"cells": 1.5}, {"cells": True}, {"seed": -1},
    {"alphas": []}, {"alphas": [-1]}, {"alphas": [1, 1]},
    {"sigma_mus": [float("nan")]}, {"sigma_rs": [float("inf")]},
    {"config": sweep.notebook_config(n_killers=1.5)},
    {"config": sweep.notebook_config(n_targets=0)},
    {"config": sweep.notebook_config(duration=float("nan"))},
    {"config": sweep.notebook_config(record_every=0)}, {"workers": 0},
])
def test_invalid_inputs_leave_no_output(tmp_path, tracked_simulator, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        sweep.run_sweep(output, **kwargs)
    assert not output.exists()
    assert not tracked_simulator


def test_tiny_real_simulator(tmp_path):
    output = sweep.run_sweep(
        tmp_path / "real", cells=3, seed=17, alphas=[2.5], sigma_mus=[2],
        sigma_rs=[5],
        config=sweep.notebook_config(n_killers=2, n_targets=3, duration=0.04, record_every=1),
    )
    summary = pd.read_csv(output / "summary.csv")
    condition = (output / summary.iloc[0].kills_data_path).parent
    pooled = pd.read_csv(condition / "kills_data.csv")
    assert len(pooled) == 3
    assert pooled.kills.ge(0).all()
    assert np.all(pooled.exposure == 0.04)
    for run in sorted((condition / "runs").iterdir()):
        raw = pd.read_csv(run / "kills_data.csv")
        snapshots = pd.read_csv(run / "snapshots.csv.gz")
        events = pd.read_csv(run / "events.csv.gz")
        assert len(raw) == 2
        assert snapshots.time.min() == 0
        assert snapshots.time.max() == pytest.approx(0.04)
        assert raw.kills.sum() == (events.event == "target_killed").sum()
        assert events.empty and set(events) == {"time", "step", "event", "killer_id", "target_id", "value", "reason"}


@pytest.mark.parametrize("stride", [1, 3])
def test_streamed_history_matches_in_memory_with_tied_events_and_shared_damage(tmp_path, stride):
    config = sweep.notebook_config(n_killers=2, n_targets=2, duration=0.19, record_every=stride)
    config = replace(config, seed=33, mechanics=replace(config.mechanics, binding_rate=1e6,
                     unbinding_rate=0, adhesion_kt=0, repulsion_kt=0))
    initial = dict(initial_killer_positions=np.array([[-24., 0.], [24., 0.]]),
                   initial_target_positions=np.array([[0., 0.], [300., 0.]]))
    reference = Simulation(config).run(**initial)
    with sweep.history_sink(tmp_path) as sink:
        streamed = Simulation(config).run(step_sink=sink, **initial)
    pd.testing.assert_frame_equal(streamed.events, reference.events, check_exact=True)
    pd.testing.assert_frame_equal(streamed.final_cells, reference.final_cells, check_exact=True)
    pd.testing.assert_frame_equal(streamed.snapshots, reference.final_cells, check_exact=True)
    assert reference.events.time.eq(0).sum() >= 4
    assert reference.kills_per_cell().sum() == 1

    snapshots = pd.read_csv(tmp_path / "snapshots.csv.gz", float_precision="round_trip")
    expected = pd.read_csv(io.StringIO(reference.snapshots.to_csv(index=False)), float_precision="round_trip")
    pd.testing.assert_frame_equal(snapshots, expected, check_exact=True)
    events = pd.read_csv(tmp_path / "events.csv.gz", float_precision="round_trip")
    expected = pd.read_csv(io.StringIO(reference.events.to_csv(index=False)), float_precision="round_trip")
    pd.testing.assert_frame_equal(events, expected, check_exact=True)
    steps = pd.read_csv(tmp_path / "steps.csv.gz", float_precision="round_trip")
    assert steps.step.tolist() == list(range(len(steps)))
    assert steps.time.iloc[0] == steps.dt.iloc[0] == 0
    assert steps.time.iloc[-1] == config.duration
    np.testing.assert_allclose(steps.time.diff().iloc[1:], steps.dt.iloc[1:])
    assert steps.dt.iloc[1:].gt(0).all()
    assert steps.targets_alive.iloc[0] == 2 and steps.targets_alive.iloc[-1] == 1
    assert steps.productive_synapses.max() == 2
    assert snapshots.groupby("step").size().eq(4).all()
    expected_steps = sorted(set(steps.step[::stride]) | {steps.step.iloc[-1]})
    assert snapshots.step.unique().tolist() == expected_steps
    damage = pd.read_csv(tmp_path / "pair_damage.csv.gz", float_precision="round_trip")
    assert set(damage.killer_id) == {0, 1} and set(damage.target_id) == {0}
    np.testing.assert_allclose(damage.groupby("killer_id").damage_increment.cumsum(), damage.cumulative_damage)
    death = reference.events.loc[reference.events.event.eq("target_killed")].iloc[0]
    assert damage.step.max() == death.step


def test_restart_verifies_committed_runs_and_recovers_partial_simulations(tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    cfg = sweep.notebook_config(n_killers=2, n_targets=3, duration=0.08)
    kwargs = dict(cells=3, config=cfg, alphas=[2.5], sigma_mus=[2], sigma_rs=[5])
    original = sweep.simulate
    seeds = []

    def fail_once(config, **kwargs):
        seeds.append(config.seed)
        result = original(config, **kwargs)
        if len(seeds) == 2:
            raise RuntimeError("interrupted before publishing")
        return result

    monkeypatch.setattr(sweep, "simulate", fail_once)
    with pytest.raises(RuntimeError, match="interrupted before publishing"):
        sweep.run_sweep(output, **kwargs)
    first, = output.glob("*/runs/run_0001")
    original_mtime = (first / "snapshots.csv.gz").stat().st_mtime_ns
    assert not (first.parent / "run_0002").exists()
    assert (first.parent / ".run_0002.partial").is_dir()
    assert sweep.read_json(output / "manifest.json")["status"] == "interrupted"
    with pytest.raises(ValueError, match="base_seed differs"):
        sweep.run_sweep(output, resume=True, seed=123, **kwargs)

    sweep.run_sweep(output, resume=True, **kwargs)
    assert seeds[1:] == [seeds[1]] * 2 and len(seeds) == 3
    assert (first / "snapshots.csv.gz").stat().st_mtime_ns == original_mtime
    assert not (first.parent / ".run_0002.partial").exists()
    assert sweep.read_json(output / "manifest.json")["completed_runs"] == 2
    assert sweep.read_json(output / "manifest.json")["status"] == "complete"

    damaged = first / "snapshots.csv.gz"
    data = damaged.read_bytes()
    damaged.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="Checksum mismatch.*snapshots"):
        sweep.run_sweep(output, resume=True, **kwargs)
    assert len(seeds) == 3 and sweep.read_json(output / "manifest.json")["status"] != "complete"


def test_transported_launcher_parallel_serial_and_archived_restart(tmp_path):
    bundle = tmp_path / "standalone bundle"
    shutil.copytree(sweep.BUNDLE, bundle, ignore=shutil.ignore_patterns(".*", "__pycache__", "outputs", "runs"))
    python = bundle / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    env = dict(os.environ, PYTHONPATH="", SWEEP_RUNS_DIR=str(tmp_path / "jobs"))
    env.pop("SWEEP_PYTHON", None)
    cmd = ["bash", str(bundle / "run_sweep.sh")]

    def launch(*args, check=True):
        return subprocess.run(cmd + list(args), cwd=tmp_path, env=env, check=check,
                              capture_output=True, text=True, timeout=60)

    plan = launch("--dry-run").stdout
    assert "1000 conditions x 20 simulations = 20000 simulations; 1000 observations/condition" in plan
    assert "50 killers + 200 targets, duration=100, cell snapshot every 1 step(s)" in plan
    assert "8 simulations" in launch("--dry-run", "--pilot").stdout
    assert launch("--workers", "0", check=False).returncode
    assert not (tmp_path / "jobs").exists()
    settings = ["--alpha", "1.25", "2.5", "--sigma-mu", "0.8", "--sigma-r", "5", "--cells", "3", "--killers-per-run",
                "2", "--targets-per-run", "3", "--duration", "0.16"]
    launch(*settings, "--workers", "2")
    parallel, = (tmp_path / "jobs").iterdir()
    assert "Finished:" in (parallel / "run.log").read_text()
    assert (parallel / "environment.txt").is_file()
    assert (parallel / "source/src/mantishrimp/simulation.py").is_file()
    serial = tmp_path / "serial"
    subprocess.run([str(python), str(bundle / "run_sweep.py"), "--output-dir", str(serial), *settings],
                   cwd=tmp_path, env=env, check=True, capture_output=True, timeout=60)
    data = parallel / "data"
    inventory = dict(line.split("  ", 1)[::-1] for line in (data / "SHA256SUMS").read_text().splitlines())
    expected = {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    assert inventory.keys() == expected - {".sweep.lock", "SHA256SUMS"}
    assert all(sweep.sha256(data / name) == digest for name, digest in inventory.items())
    for file in serial.rglob("*.csv*"):
        other = data / file.relative_to(serial)
        if file.suffix == ".gz":
            assert gzip.decompress(file.read_bytes()) == gzip.decompress(other.read_bytes())
        else:
            assert file.read_bytes() == other.read_bytes()
    # A later upload cannot change the archived code used by a resumed job.
    (bundle / "timeline.py").write_text("raise AssertionError('wrong source')\n")
    finished = sweep.read_json(data / "manifest.json")["finished_at_utc"]
    assert "4 verified, 0 simulations to run" in launch("--resume", str(parallel)).stdout
    assert sweep.read_json(data / "manifest.json")["finished_at_utc"] == finished
    assert launch("--resume", str(parallel), "--seed", "7", check=False).returncode
