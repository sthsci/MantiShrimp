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
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    columns = ["k_bind", "k_unbind", "mean_killing_efficiency", "n_runs", "obs_time"]
    if frame.empty or not set(columns).issubset(frame.columns):
        parser.error(f"summary must contain nonempty columns: {', '.join(columns)}")
    try:
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        parser.error(f"summary contains nonnumeric values: {error}")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        parser.error("rates, mean killing fractions, replicate counts and durations must be finite")
    durations = frame["obs_time"]
    if (durations <= 0).any() or durations.nunique() != 1:
        parser.error("all grid points must have the same positive observation time")
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
    mesh = ax.imshow(grid.to_numpy(), origin="lower", cmap="coolwarm",
                     vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(len(grid.columns)), [f"{x:g}" for x in grid.columns],
                  rotation=45, ha="right")
    ax.set_yticks(np.arange(len(grid.index)), [f"{x:g}" for x in grid.index])
    ax.set_xlabel(r"Binding rate $k_{\mathrm{on}}$ (min$^{-1}$)")
    ax.set_ylabel(r"Unbinding rate $k_{\mathrm{off}}$ (min$^{-1}$)")
    ax.set_title(f"Cytotoxicity at {durations.iloc[0]:g} min")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean fraction of targets killed")
    fig.text(0.5, 0.02,
             f"Parameter grid: equal-width cells; {int(counts.iloc[0])} replicates per point",
             ha="center", fontsize=9, color="#4D4D4D")
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=1.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"cytotoxicity_heatmap.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"Saved {path.resolve()}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
