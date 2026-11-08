"""Plot mean target elimination from a completed binding/unbinding sweep."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=100.0,
                        help="Simulation duration in minutes (default: 100).")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")

    frame = pd.read_csv(args.summary)
    columns = ["k_bind", "k_unbind", "mean_killing_efficiency", "n_runs"]
    if frame.empty or not set(columns).issubset(frame.columns):
        parser.error(f"summary must contain nonempty columns: {', '.join(columns)}")
    try:
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        parser.error(f"summary contains nonnumeric values: {error}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        parser.error("rates, mean killing fractions and replicate counts must be finite")
    if (frame[["k_bind", "k_unbind"]] <= 0).any().any():
        parser.error("binding and unbinding rates must be positive")
    if not frame["mean_killing_efficiency"].between(0, 1).all():
        parser.error("mean killing fractions must lie between 0 and 1")
    counts = frame["n_runs"]
    if (counts <= 0).any() or (counts % 1 != 0).any() or counts.nunique() != 1:
        parser.error("all grid points must have the same positive integer replicate count")
    if frame.duplicated(["k_bind", "k_unbind"]).any():
        parser.error("summary contains duplicate parameter combinations")
    grid = frame.pivot(index="k_unbind", columns="k_bind",
                       values="mean_killing_efficiency").sort_index().sort_index(axis=1)
    if grid.isna().any().any():
        parser.error("summary is missing parameter combinations; refusing to plot missing data as zero")
    if min(grid.shape) < 2:
        parser.error("a smooth heatmap needs at least two rates on each axis")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    max_mean = float(grid.to_numpy().max())
    # Keep a valid colour range when every grid point has zero kills.
    color_max = max_mean if max_mean > 0 else 1.0
    mesh = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(),
                         shading="gouraud", cmap="viridis", vmin=0,
                         vmax=color_max, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Binding rate $k_{\mathrm{on}}$ (min$^{-1}$)")
    ax.set_ylabel(r"Unbinding rate $k_{\mathrm{off}}$ (min$^{-1}$)")
    ax.set_title(f"Cytotoxicity at {args.duration:g} min")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04,
                 ticks=np.linspace(0, color_max, 6) if max_mean > 0 else [0],
                 label="Mean fraction of targets killed")
    fig.text(0.5, 0.02,
             f"{len(grid.columns)} × {len(grid.index)} sampled points; "
             f"{int(counts.iloc[0])} replicates per point\n"
             "Log10 axes; Gouraud colour interpolation between samples",
             ha="center", fontsize=9, color="#4D4D4D")
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=1.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"cytotoxicity_heatmap.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"Saved {path.resolve()}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
