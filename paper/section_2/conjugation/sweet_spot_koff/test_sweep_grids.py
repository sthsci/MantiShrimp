"""Small local regression checks for independently configured sweep grids."""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "sweep_under_test", Path(__file__).with_name("10x10.py")
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepGridTests(unittest.TestCase):
    def test_default_grids_are_log_spaced_and_keep_server_ranges(self):
        for values, endpoints in [(sweep.K_BIND_GRID, [0.1, 25]),
                                  (sweep.K_UNBIND_GRID, [0.005, 30])]:
            self.assertEqual(len(values), 21)
            np.testing.assert_allclose(values[[0, -1]], endpoints)
            log_steps = np.diff(np.log10(values))
            np.testing.assert_allclose(log_steps, log_steps[0])

    def run_tiny_sweep(self, extra_args=()):
        original_config = sweep.make_config

        def tiny_config(k_bind, k_unbind, seed, smoke=False):
            return replace(original_config(k_bind, k_unbind, seed, smoke=True),
                           duration=0.02)

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.object(sweep, "K_BIND_GRID", [0.1, 0.3]), \
                 patch.object(sweep, "K_UNBIND_GRID", [0.02, 0.2, 2]), \
                 patch.object(sweep, "make_config", side_effect=tiny_config), \
                 patch.object(sys, "argv", ["10x10.py", "--output-dir", directory,
                                            *extra_args]), redirect_stdout(output):
                sweep.main()
            raw = pd.read_csv(Path(directory) / f"{sweep.PREFIX}_HPC.csv.gz")
            summary = pd.read_csv(Path(directory) / f"{sweep.PREFIX}_summary.csv")
        return raw, summary, output.getvalue()

    def test_asymmetric_grids_form_full_cartesian_product(self):
        raw, summary, output = self.run_tiny_sweep()
        columns = ["k_bind", "k_unbind", "seed"]
        expected = {(on, off, seed) for on in [0.1, 0.3]
                    for off in [0.02, 0.2, 2] for seed in range(1, 6)}
        self.assertEqual(len(raw), 30)
        self.assertEqual(set(raw[columns].itertuples(index=False, name=None)), expected)
        self.assertFalse(raw.duplicated(columns).any())
        self.assertEqual(len(summary), 6)
        self.assertTrue(summary["n_runs"].eq(5).all())
        self.assertIn("[30/30]", output)

    def test_smoke_mode_still_runs_one_combination(self):
        raw, summary, output = self.run_tiny_sweep(["--smoke-test"])
        self.assertEqual(raw[["k_bind", "k_unbind", "seed"]].values.tolist(),
                         [[0.5, 0.5, 1]])
        self.assertEqual(summary["n_runs"].tolist(), [1])
        self.assertIn("[1/1]", output)

    def test_invalid_grids_fail_before_simulation(self):
        invalid = [[], [0], [-1], [float("nan")], [float("inf")], [0.1, 0.1]]
        for name in ["K_BIND_GRID", "K_UNBIND_GRID"]:
            for values in invalid:
                with self.subTest(grid=name, values=values), \
                     patch.object(sweep, name, values), \
                     patch.object(sweep, "Simulation") as simulation, \
                     patch.object(sys, "argv", ["10x10.py"]), \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        sweep.main()
                    self.assertEqual(error.exception.code, 2)
                    simulation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
