"""Regression checks for log coordinates, source values and incomplete grids."""

import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

import plot_heatmap


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(k_bind=x, k_unbind=y, mean_killing_efficiency=0.1*i + 0.2*j,
                 n_runs=5)
            for j, y in enumerate([30, 0.005])
            for i, x in enumerate([25, 0.1, 0.5])
        ])

    def test_log_axes_preserve_irregular_grid_values_and_export(self):
        for all_zero in [False, True]:
            with self.subTest(all_zero=all_zero), tempfile.TemporaryDirectory() as directory:
                frame = self.frame.copy()
                if all_zero:
                    frame["mean_killing_efficiency"] = 0
                summary = Path(directory)/"summary.csv"
                frame.to_csv(summary, index=False)
                original = summary.read_bytes()
                with patch.object(sys, "argv", ["plot_heatmap.py", str(summary),
                                                "--output-dir", directory]), \
                     patch.object(plot_heatmap.plt, "close"), redirect_stdout(io.StringIO()):
                    plot_heatmap.main()
                fig = plot_heatmap.plt.gcf()
                try:
                    ax = fig.axes[0]
                    self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
                    np.testing.assert_allclose(ax.get_xlim(), [0.1, 25])
                    np.testing.assert_allclose(ax.get_ylim(), [0.005, 30])
                    self.assertEqual(ax.get_title(), "Cytotoxicity at 100 min")
                    mesh = ax.collections[0]
                    expected = frame.pivot(index="k_unbind", columns="k_bind",
                                           values="mean_killing_efficiency")
                    np.testing.assert_allclose(mesh.get_array(), expected.to_numpy())
                    np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], expected.columns)
                    np.testing.assert_allclose(mesh.get_coordinates()[:, 0, 1], expected.index)
                    self.assertEqual(mesh.get_clim(), (0, 1 if all_zero else 0.4))
                    self.assertEqual(mesh.cmap.name, "viridis")
                    self.assertIn("interpolation", fig.texts[0].get_text())
                    for extension in ["png", "pdf"]:
                        self.assertGreater((Path(directory)/f"cytotoxicity_heatmap.{extension}").stat().st_size, 1000)
                    self.assertEqual(summary.read_bytes(), original)
                finally:
                    plot_heatmap.plt.close(fig)

    def test_incomplete_or_invalid_data_is_rejected(self):
        invalid_frames = [
            self.frame.iloc[:-1],
            self.frame.assign(k_bind=0),
            self.frame.assign(k_unbind=np.nan),
            self.frame.assign(mean_killing_efficiency=1.1),
            self.frame.assign(n_runs=[1, 5, 5, 5, 5, 5]),
            pd.concat([self.frame, self.frame.iloc[:1]]),
            self.frame.iloc[:3],
        ]
        for frame in invalid_frames:
            with self.subTest(frame=frame.to_dict()), tempfile.TemporaryDirectory() as directory:
                summary = Path(directory)/"summary.csv"
                frame.to_csv(summary, index=False)
                with patch.object(sys, "argv", ["plot_heatmap.py", str(summary),
                                                "--output-dir", directory]), \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        plot_heatmap.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((Path(directory)/"cytotoxicity_heatmap.png").exists())


if __name__ == "__main__":
    unittest.main()
