import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from assay import make_config
from plot_sweep import plot
from sweep import AXES, collect, make_plan, task_file


class DispersionSweepTest(unittest.TestCase):
    def setUp(self):
        self.settings = dict(k_on=5.0, k_off=[0.01, 0.1, 1, 10], adhesion_kt=[0, 10, 50, 300],
                             seeds=[1, 2, 3, 4, 5], spatial_seed=66, sigma_mu=0.8, reuse_reference=True)

    def test_agreed_grid_and_reference_totals(self):
        plan, old = make_plan(self.settings)
        self.assertEqual(len(plan), 60)
        self.assertEqual(plan.groupby(AXES).seed.agg(list).tolist(), [[1, 2, 3, 4, 5]] * 12)
        self.assertEqual(old.adhesion_kt.tolist(), [0, 10, 50, 300])
        self.assertEqual(old.total_killed.tolist(), [249, 257, 158, 129])
        self.assertEqual(old.n_seeds.tolist(), [5] * 4)
        self.assertTrue(old.sigma_mu.eq(0.8).all())
        task = next(plan.itertuples(index=False))
        config = make_config(task)
        self.assertEqual((config.mechanics.unbinding_rate, config.mechanics.adhesion_kt),
                         (task.k_off, task.adhesion_kt))
        self.assertEqual((config.mechanics.binding_rate, config.duration, config.n_targets,
                          config.n_killers, config.initial_separation_factor), (5, 100, 100, 25, 2.5))

    def test_killer_noise_three_runs_every_seed(self):
        settings = {**self.settings, "sigma_mu": 3}
        plan, old = make_plan(settings)
        self.assertEqual(plan.groupby(AXES).seed.agg(list).tolist(), [[1, 2, 3, 4, 5]] * 16)
        self.assertTrue(old.empty)
        self.assertTrue(plan.sigma_mu.eq(3).all())
        motility = make_config(next(plan.itertuples(index=False))).motility
        self.assertEqual((motility.killer_polarity_noise, motility.target_polarity_noise), (3, 0.2))
        with self.assertRaisesRegex(ValueError, "sigma_mu"):
            make_plan({**settings, "sigma_mu": -1})

    def test_incompatible_runs_do_not_reuse_baseline(self):
        for override in ({"spatial_seed": 67}, {"seeds": [6, 7, 8, 9, 10]},
                         {"k_on": 2}, {"reuse_reference": False}):
            with self.subTest(override=override):
                plan, old = make_plan({**self.settings, **override})
                self.assertEqual(len(plan), 80)
                self.assertTrue(old.empty)
        with self.assertRaises(ValueError):
            make_plan({**self.settings, "k_off": [0.01, 0.01]})

    def test_plot_rejects_missing_cell(self):
        _, old = make_plan(self.settings)
        incomplete = pd.concat([old, old.iloc[:-1].assign(k_off=1.0)])
        with TemporaryDirectory() as output, self.assertRaisesRegex(ValueError, "incomplete"):
            plot(incomplete, Path(output))

    def test_collector_keeps_archived_means_separate_from_new_replicates(self):
        plan, _ = make_plan(self.settings)
        cwd = Path.cwd()
        with TemporaryDirectory() as output, patch("plot_sweep.plot"):
            os.chdir(output)
            try:
                Path("settings.json").write_text(json.dumps(self.settings))
                with self.assertRaisesRegex(ValueError, "missing task IDs"):
                    collect(plan)
                Path("tasks").mkdir()
                for task in plan.itertuples(index=False):
                    pd.DataFrame([{**task._asdict(), "targets_killed": task.seed,
                                   "fraction_targets_killed": task.seed / 100}]).to_csv(
                                       task_file(task.task), index=False)
                collect(plan)
                summary = pd.read_csv("summary.csv")
                self.assertEqual(len(pd.read_csv("replicates.csv")), 60)
                self.assertEqual(len(summary), 16)
                self.assertTrue(summary.loc[summary.source == "new", "mean_fraction_killed"].eq(.03).all())
                self.assertEqual(summary.query("k_off == 0.01 and adhesion_kt == 10")
                                 .mean_fraction_killed.item(), .514)
            finally:
                os.chdir(cwd)
