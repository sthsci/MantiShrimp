"""Check the portable sweep against the notebook's executable settings."""

import ast
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("notebook_sweep", DIRECTORY / "10x10.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def notebook_reference():
    """Execute only configuration, placement, and assertions; omit all plotting."""
    notebook = json.loads((DIRECTORY / "parameter_test.ipynb").read_text())
    cells = [ast.parse("".join(cell["source"])) for cell in notebook["cells"]
             if cell["cell_type"] == "code"]
    namespace = {name: getattr(sweep, name) for name in (
        "SimulationConfig", "DomainConfig", "MechanicsConfig", "MotilityConfig",
        "CytotoxicityConfig", "OutputConfig",
    )}
    namespace["np"] = np
    config_cell = next(cell for cell in cells if any(
        isinstance(node, ast.FunctionDef) and node.name == "make_config"
        for node in cell.body
    ))
    config_nodes = [node for node in config_cell.body if
                    isinstance(node, ast.FunctionDef) or
                    (isinstance(node, ast.Assign) and
                     isinstance(node.targets[0], ast.Name) and
                     node.targets[0].id in {"K_BIND", "K_UNBIND", "SEED", "SMOKE"})]
    exec(compile(ast.Module(body=config_nodes, type_ignores=[]), "notebook-config", "exec"),
         namespace)
    namespace["config"] = namespace["make_config"](
        namespace["K_BIND"], namespace["K_UNBIND"], namespace["SEED"])
    layout_cell = next(cell for cell in cells if any(
        isinstance(node, ast.FunctionDef) and node.name == "make_initial_positions"
        for node in cell.body
    ))
    exec(compile(layout_cell, "notebook-layout", "exec"), namespace)
    # The preview cell validates the layout before its first plotting statement.
    checks = next(cell for cell in cells if any(
        isinstance(node, ast.Assert) and "target_distances" in ast.unparse(node)
        for node in cell.body
    ))
    check_nodes = []
    for node in checks.body:
        if any(isinstance(item, ast.Name) and item.id == "plt" for item in ast.walk(node)):
            break
        check_nodes.append(node)
    exec(compile(ast.Module(body=check_nodes, type_ignores=[]), "notebook-checks", "exec"),
         namespace)
    return namespace


class NotebookSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = notebook_reference()

    def test_all_config_fields_match_notebook_including_defaults(self):
        reference_config = self.reference["make_config"]
        notebook_run = (self.reference["K_BIND"], self.reference["K_UNBIND"], self.reference["SEED"])
        for smoke in (False, True):
            for rates in (notebook_run, (0.03, 10.0, 2)):
                with self.subTest(smoke=smoke, rates=rates):
                    self.assertEqual(sweep.make_config(*rates, smoke=smoke).to_dict(),
                                     reference_config(*rates, smoke=smoke).to_dict())

    def test_seed_42_layout_and_polarities_match_notebook(self):
        ref = self.reference
        self.assertEqual(sweep.SPATIAL_SEED, ref["SEED"])
        for name in ("TARGET_CLUSTER_RADIUS", "KILLER_RING_INNER_RADIUS",
                     "KILLER_RING_OUTER_RADIUS", "INITIAL_KILLER_POLARITY_MAGNITUDE"):
            self.assertEqual(getattr(sweep, name), ref[name])
        killers, targets = sweep.make_initial_positions(
            ref["config"], sweep.TARGET_CLUSTER_RADIUS,
            sweep.KILLER_RING_INNER_RADIUS, sweep.KILLER_RING_OUTER_RADIUS)
        np.testing.assert_array_equal(killers, ref["initial_killer_positions"])
        np.testing.assert_array_equal(targets, ref["initial_target_positions"])
        np.testing.assert_array_equal(sweep.make_initial_polarities(killers),
                                      ref["initial_killer_polarities"])

    def test_bundled_engine_records_requested_initial_polarities(self):
        ref = self.reference
        config = replace(ref["config"], duration=0.01)
        result = sweep.Simulation(config).run(**{
            name: ref[name] for name in ("initial_killer_positions", "initial_target_positions",
                                        "initial_killer_polarities")
        })
        initial = result.snapshots.loc[result.snapshots["time"].eq(0)]
        killers = initial.loc[initial["cell_type"].eq("killer")]
        targets = initial.loc[initial["cell_type"].eq("target")]
        np.testing.assert_array_equal(killers[["polarity_x", "polarity_y"]],
                                      ref["initial_killer_polarities"])
        np.testing.assert_array_equal(targets[["polarity_x", "polarity_y"]], 0)
        self.assertAlmostEqual(result.duration, config.duration)

    def test_main_reuses_notebook_layout_across_rates_and_dynamics_seeds(self):
        with patch.object(sweep, "K_BIND_GRID", [0.1, 5]), \
             patch.object(sweep, "K_UNBIND_GRID", [0.1]), \
             patch.object(sweep, "SEEDS", [1, 2]), \
             patch.object(sweep, "Simulation") as simulation, \
             patch.object(sweep, "summarize", return_value=({}, [])), \
             patch.object(sweep, "save_outputs"), \
             patch.object(sys, "argv", ["10x10.py"]), redirect_stdout(io.StringIO()):
            sweep.main()
        self.assertEqual([call.args[0].seed for call in simulation.call_args_list],
                         [1, 2, 1, 2])
        self.assertEqual(simulation.return_value.run.call_count, 4)
        for call in simulation.return_value.run.call_args_list:
            for name in ("initial_killer_positions", "initial_target_positions",
                         "initial_killer_polarities"):
                np.testing.assert_array_equal(call.kwargs[name], self.reference[name])


if __name__ == "__main__":
    unittest.main()
