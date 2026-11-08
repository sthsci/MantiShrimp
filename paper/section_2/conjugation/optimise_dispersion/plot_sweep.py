"""Draw the measured dispersion grid and its unbinding-rate profiles from summary.csv."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def plot(frame, directory):
    axes = ["k_off", "adhesion_kt"]
    conditions = ["k_on", "duration", "spatial_seed", "n_killers", "n_targets", "n_seeds"]
    if "sigma_mu" in frame:
        conditions.append("sigma_mu")
    mean, sd = "mean_fraction_killed", "sd_fraction_killed"
    required = axes + conditions + [mean, sd]
    if frame.empty or not set(required).issubset(frame):
        raise ValueError(f"Expected a nonempty summary with columns: {', '.join(required)}")
    if not np.isfinite(frame[axes + conditions + [mean]].to_numpy(dtype=float)).all():
        raise ValueError("Grid coordinates, means and assay settings must be finite")
    if frame.duplicated(axes).any() or len(frame[conditions].drop_duplicates()) != 1:
        raise ValueError("Expected unique pairs from one assay with equal replication")
    n = frame.n_seeds.iloc[0]
    if (n < 1 or n % 1 or (frame[axes] < 0).any().any() or not frame[mean].between(0, 1).all()
            or (n > 1 and (not np.isfinite(frame[sd]).all() or (frame[sd] < 0).any()))):
        raise ValueError("Invalid coordinates, replicate count, mean fraction or sample SD")
    grid = frame.pivot(index="k_off", columns="adhesion_kt", values=mean).sort_index().sort_index(axis=1)
    if min(grid.shape) < 2 or grid.isna().any().any():
        raise ValueError("Cannot plot an incomplete two-dimensional grid")

    fig, (ax, lines) = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    mesh = ax.imshow(grid, origin="lower", interpolation="nearest", aspect="auto",
                     cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(grid.columns)), [f"{a:g}" for a in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{k:g}" for k in grid.index])
    ax.set(xlabel="Killer–target adhesion (model units)", ylabel=r"$k_{\rm off}$ (min$^{-1}$)",
           title="Mean targets killed")
    if grid.size <= 100:
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            ax.text(j, i, f"{value:.1%}", ha="center", va="center",
                    color="black" if value > 0.6 else "white")
    fig.colorbar(mesh, ax=ax, format=PercentFormatter(1), shrink=0.85)
    for adhesion, rows in frame.sort_values("k_off").groupby("adhesion_kt"):
        lines.errorbar(rows.k_off, rows[mean], yerr=rows[sd] if n > 1 else None,
                       marker="o", capsize=3, label=f"{adhesion:g}")
    if min(grid.index) > 0:
        lines.set_xscale("log")
    lines.set_xticks(grid.index, [f"{k:g}" for k in grid.index])
    lines.yaxis.set_major_formatter(PercentFormatter(1))
    lines.set(ylim=(-0.02, 1.02), xlabel=r"$k_{\rm off}$ (min$^{-1}$)", ylabel="Targets killed",
              title="Mean ± 1 SD across seeds" if n > 1 else "Single replicate per pair")
    lines.legend(title="Adhesion", frameon=False)
    first = frame.iloc[0]
    noise_label = rf"killer $\sigma_\mu$ = {first.sigma_mu:g}   |   " if "sigma_mu" in frame else ""
    fig.suptitle(f"Dispersed targets at {first.duration:g} min   |   "
                 f"$k_{{\\rm on}}$ = {first.k_on:g} min⁻¹   |   "
                 f"{noise_label}layout {first.spatial_seed:g}, {n:g} seed(s) per pair")
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"dispersion_response.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, help="defaults to the summary directory")
    args = parser.parse_args()
    plot(pd.read_csv(args.summary), args.output_dir or args.summary.parent)
