"""Run with the ABM Python environment; checks the notebook without Jupyter."""

import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import numpy as np
from PIL import Image


def check():
    notebook = Path(__file__).resolve().with_name("demo_new.ipynb")
    root = notebook.parents[2]
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    }
    original_cwd, original_path = Path.cwd(), sys.path.copy()
    try:
        for directory in (root, notebook.parent):
            os.chdir(directory)
            sys.path[:] = original_path
            for module in ("simulator", "assistant_function", "virsualisation"):
                sys.modules.pop(module, None)
            scope = {}
            for name in ("setup", "parameters"):
                exec(compile(cells[name], f"{notebook.name}:{name}", "exec"), scope)
            assert scope["PROJECT_ROOT"] == root
            assert set(scope["params"]) == set(inspect.signature(scope["KTSimulator"]).parameters)
            derived = {"cell_history_df", "killer_positions", "target_positions",
                       "killer_radius", "target_radius", "box", "boundary_condition", "ax_box"}
            controls = set(inspect.signature(scope["animate_KT"]).parameters) - derived
            assert set(scope["animation_params"]) <= controls
            assert controls - set(scope["animation_params"]) <= {"traj_linewidth", "traj_alpha", "cir_linewidth"}
            scope["params"].update(N_KILLER=2, N_TARGET=3, SIM_DURATION=0.12, DT0=0.04,
                                   BOX_X_MIN=-100, BOX_X_MAX=100, BOX_Y_MIN=-100,
                                   BOX_Y_MAX=100, init_cell_bound=(-90, 90, -90, 90),
                                   min_distance_factor=1.1)
            run_code = compile(cells["run-simulation"], "demo_new.ipynb:run-simulation", "exec")
            exec(run_code, scope)
            df = scope["cell_history_df"].copy(deep=True)
            steps = df["step"].nunique()
            assert steps >= 2 and len(df) == steps * 5
            expected = {"killer_positions": (steps, 2, 2), "target_positions": (steps, 3, 2),
                        "killingProbability_history": (steps, 2), "DeathFactor_history": (steps, 3)}
            first = {name: np.asarray(scope[name]).copy() for name in expected}
            for name, shape in expected.items():
                assert first[name].shape == shape and np.isfinite(first[name]).all(), name
            np.random.seed(123456)
            exec(run_code, scope)
            assert df.equals(scope["cell_history_df"]), "Repeated seeds changed cell history"
            for name in first:
                np.testing.assert_array_equal(first[name], scope[name])
            scope["params"]["RNG_SEED"] += 1
            exec(run_code, scope)
            assert not np.array_equal(first["killer_positions"], scope["killer_positions"])
            with TemporaryDirectory() as temporary:
                gif = Path(temporary) / "preview.gif"
                scope["animation_params"].update(save_gif_path=str(gif), render_html=True,
                                                 frame_step=1, traj_steps=2, fig_dpi=40,
                                                 figsize=(5, 4), gif_fps=5,
                                                 traj_linewidth=1.8, traj_alpha=0.25,
                                                 dead_target_radius=3, cir_linewidth=1.2,
                                                 legend_labels=["Killer", "Alive target", "Dead target"])
                for boundary in ("periodic", "confined"):
                    scope["sim_settings"]["boundary_condition"] = boundary
                    with patch("virsualisation.display") as display:
                        exec(compile(cells["animate"], "demo_new.ipynb:animate", "exec"), scope)
                    assert display.called and "data:image/png;base64," in display.call_args.args[0].data
                    with Image.open(gif) as preview:
                        assert preview.format == "GIF" and preview.n_frames >= 2
                    for step in (1, 2, 1):  # Periodic trails are recreated on every frame.
                        lines = [a for a in scope["animation"]._func(step) if isinstance(a, Line2D)]
                        assert lines
                        assert all(a.get_linewidth() == 1.8 and a.get_alpha() == 0.25 for a in lines)
                        assert min(c.get_zorder() for c in scope["animation"]._fig.axes[0].patches) > max(a.get_zorder() for a in lines)
                    fig = scope["animation"]._fig
                    assert len(fig.axes) == 1 and len(fig.legends) == 1
                    legend = fig.legends[0]
                    assert [text.get_text() for text in legend.get_texts()] == scope["animation_params"]["legend_labels"]
                    radii = [scope["sim_settings"]["KILLER_RADIUS"], scope["sim_settings"]["TARGET_RADIUS"], 3]
                    for handle, radius, colour in zip(legend.legend_handles, radii,
                            (scope["animation_params"][key] for key in ("killer_colour", "live_target_colour", "dead_target_colour"))):
                        assert handle.get_marker() == "o" and handle.get_markersize() == 12 * radius / max(radii)
                        assert to_rgba(handle.get_markerfacecolor()) == to_rgba(colour)
                        assert handle.get_markeredgewidth() == 1.2
                    fig.canvas.draw()
                    legend_box = legend.get_window_extent()
                    assert legend_box.y1 < fig.axes[0].get_window_extent().y0
                    assert legend_box.x0 >= 0 and legend_box.x1 <= fig.bbox.width
                    assert legend_box.y0 >= 0
                    svg = Path(temporary) / "frame.svg"
                    with scope["plt"].rc_context({"svg.fonttype": "none"}):
                        fig.savefig(svg, format="svg")
                    assert all(label in svg.read_text() for label in scope["animation_params"]["legend_labels"])
            scope["params"]["boundary_condition"] = "confined"
            exec(run_code, scope)
            for name, radius in (("killer_positions", "KILLER_RADIUS"), ("target_positions", "TARGET_RADIUS")):
                assert (np.abs(scope[name][1:]) <= 100 - scope["params"][radius] + 1e-9).all()
            for invalid in ({"DT0": 0}, {"boundary_condition": "typo"},
                            {"KILL_PROB_INIT": 1.5}, {"TARGET_STATE_INIT": np.nan}):
                valid = scope["params"].copy()
                scope["params"].update(invalid)
                try:
                    exec(run_code, scope)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Invalid input was accepted: {invalid}")
                finally:
                    scope["params"] = valid
            scope["params"].update(region_shape="circle", init_mode="gaussian",
                                   ini_cell_centre=(0, 0), ini_cell_region_radius=85)
            exec(run_code, scope)
            for name in ("killer_positions", "target_positions"):
                assert (np.linalg.norm(scope[name][0], axis=1) <= 85).all()
            scope["plt"].close("all")
            print(f"PASS: {directory.relative_to(root)} (inputs, seeds, GIF, HTML, boundaries, Gaussian circle)")
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


if __name__ == "__main__":
    check()
