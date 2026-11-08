from dataclasses import replace
import unittest

import pandas as pd

from sweep import OutputConfig, Simulation, make_config, make_plan, rank_results, report_best


SETTINGS = {"adhesion_kt": [0, 300, 600], "seeds": [1, 2], "k_on": 5.0, "k_off": 0.01}


class AdhesionSweepTest(unittest.TestCase):
    def test_only_adhesion_changes_within_a_seed(self):
        plan = make_plan(SETTINGS)
        self.assertEqual(plan.groupby("adhesion_kt").seed.agg(list).tolist(), [[1, 2]] * 3)
        tasks = list(plan[plan.seed == 1].itertuples(index=False))
        baseline = make_config(tasks[0])
        for task in tasks:
            config = make_config(task)
            mechanics = replace(baseline.mechanics, adhesion_kt=task.adhesion_kt)
            self.assertEqual(config, replace(baseline, mechanics=mechanics))
            self.assertEqual((config.mechanics.binding_rate, config.mechanics.unbinding_rate), (5.0, 0.01))
            self.assertEqual((config.n_killers, config.n_targets, config.duration, task.spatial_seed),
                             (25, 100, 100.0, 66))

    def test_sparse_recording_preserves_killing_and_events(self):
        task = next(make_plan(SETTINGS).query("adhesion_kt == 300").itertuples(index=False))
        config = replace(make_config(task), n_killers=1, n_targets=1, duration=3.0)
        initial = dict(initial_killer_positions=[[0, 0]], initial_target_positions=[[22, 0]])
        sparse = Simulation(config).run(**initial)
        dense = Simulation(replace(config, output=OutputConfig(record_every=1))).run(**initial)
        self.assertGreater(sparse.summary()["synapse_episodes"], 0)
        self.assertEqual(sparse.summary()["targets_killed"], 1)
        self.assertEqual(sparse.summary(), dense.summary())
        pd.testing.assert_frame_equal(sparse.events, dense.events)
        pd.testing.assert_frame_equal(sparse.final_cells, dense.final_cells)

    def test_ranking_keeps_ties_and_flags_boundaries(self):
        plan = make_plan(SETTINGS)
        for counts, expected, note in [
            ([10, 8, 11, 9, 7, 8], [300], "Interior candidate"),
            ([10, 8, 11, 7, 7, 8], [0, 300], "boundary"),
            ([10, 8, 11, 7, 9, 9], [0, 300, 600], "All tested means are equal"),
        ]:
            with self.subTest(expected=expected):
                rows = plan.assign(targets_killed=counts)
                rows["fraction_targets_killed"] = rows.targets_killed / rows.n_targets
                summary = rank_results(rows)
                self.assertEqual(summary.loc[summary.best_tested, "adhesion_kt"].tolist(), expected)
                self.assertIn(note, report_best(summary))
