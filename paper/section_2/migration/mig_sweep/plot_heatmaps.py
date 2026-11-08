"""Plot the two motility heatmaps from completed sweep CSVs, without simulations."""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

PARAMETERS = ["alpha", "sigma_mu", "sigma_r"]
METRIC = "mean_unique_contacts_per_killer"


def load_results(results_dir):
    results_dir = Path(results_dir)
    required = [results_dir / name for name in ("settings.json", "runs.csv.gz", "summary.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing sweep results: " + ", ".join(missing))
    settings = json.loads((results_dir / "settings.json").read_text())
    runs = pd.read_csv(results_dir / "runs.csv.gz", float_precision="round_trip")
    summary = pd.read_csv(results_dir / "summary.csv", float_precision="round_trip")
    run_keys = PARAMETERS + ["seed"]
    for frame, columns in (
        (runs, run_keys + ["unique_contacts_per_killer"]),
        (summary, PARAMETERS + [METRIC, "n_runs"]),
    ):
        if frame.empty or not set(columns).issubset(frame.columns):
            raise ValueError(f"Results must contain nonempty columns: {', '.join(columns)}")
    if runs.duplicated(run_keys).any() or summary.duplicated(PARAMETERS).any():
        raise ValueError("Duplicate parameter/seed rows in the saved results.")
    expected = pd.MultiIndex.from_product(
        [settings["grids"][name] for name in PARAMETERS] + [settings["seeds"]],
        names=run_keys,
    )
    actual = pd.MultiIndex.from_frame(runs[run_keys])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(f"Incomplete or mismatched grid: {len(runs)} rows; expected {len(expected)}.")
    expected_conditions = expected.droplevel("seed").unique()
    actual_conditions = pd.MultiIndex.from_frame(summary[PARAMETERS])
    if (len(expected_conditions.difference(actual_conditions))
            or len(actual_conditions.difference(expected_conditions))
            or not summary["n_runs"].eq(len(settings["seeds"])).all()):
        raise ValueError("The summary is incomplete or does not match settings.json.")
    values = runs["unique_contacts_per_killer"]
    means = runs.groupby(PARAMETERS)["unique_contacts_per_killer"].mean().sort_index()
    saved_means = summary.set_index(PARAMETERS)[METRIC].sort_index()
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(saved_means).all() or (saved_means < 0).any()
            or not np.allclose(means.to_numpy(), saved_means.to_numpy())):
        raise ValueError("The summary contact means do not match valid per-run results.")
    return settings, runs, summary


def heatmap_slice(frame, settings, varying, fixed, value):
    available = np.sort(frame[fixed].unique())
    nearest = available[np.argmin(np.abs(available - value))]
    if not np.isclose(nearest, value, rtol=1e-12, atol=0):
        raise ValueError(f"{fixed}={value:g} was not swept. Choose from {available.tolist()}.")
    selected = frame.loc[frame[fixed].eq(nearest)]
    matrix = selected.pivot(index=varying, columns="alpha", values=METRIC)
    matrix = matrix.reindex(index=sorted(settings["grids"][varying]),
                            columns=sorted(settings["grids"]["alpha"]))
    if matrix.isna().any().any():
        raise ValueError(f"Missing parameter combinations in the {varying} slice.")
    return matrix


def plot_heatmaps(mu_heatmap, r_heatmap, sigma_r, sigma_mu, smoke_test=False, pilot=False):
    import matplotlib.pyplot as plt

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False,
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.top": False, "ytick.right": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120,
    }
    maximum = max(mu_heatmap.to_numpy().max(), r_heatmap.to_numpy().max())
    norm = Normalize(vmin=0, vmax=maximum if maximum > 0 else 1)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), layout="constrained")
        panels = [
            (mu_heatmap, r"Polarity noise, $\sigma_\mu$", rf"Fixed $\sigma_r={sigma_r:g}$"),
            (r_heatmap, r"Translational noise, $\sigma_r$", rf"Fixed $\sigma_\mu={sigma_mu:g}$"),
        ]
        for label, ax, (matrix, ylabel, title) in zip("AB", axes, panels):
            values = matrix.to_numpy()
            im = ax.imshow(values, origin="lower", aspect="auto", cmap="Blues", norm=norm,
                           interpolation="nearest")
            ax.set_xticks(range(len(matrix.columns)), [f"{v:g}" for v in matrix.columns])
            ax.set_yticks(range(len(matrix.index)), [f"{v:g}" for v in matrix.index])
            ax.set(xlabel=r"Killer polarity decay, $\alpha$", ylabel=ylabel, title=title)
            ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=14)
            ax.grid(False)
            for (row, column), value in np.ndenumerate(values):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if norm(value) > 0.6 else "#172a3a")
        fig.colorbar(im, ax=axes, shrink=0.88, label="Unique target contacts per killer")
        fig.suptitle("Killer motility and target contacts"
                     + (" (smoke test)" if smoke_test else " (pilot)" if pilot else ""))
    return fig


def save_figure(fig, output_dir, sigma_r, sigma_mu):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"motility_contacts_sigma_r_{sigma_r:g}_sigma_mu_{sigma_mu:g}"
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved {path.resolve()}", flush=True)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing settings.json and CSVs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma-r", type=float, help="Fixed value; default is nearest grid point to 1.")
    parser.add_argument("--sigma-mu", type=float, help="Fixed value; default is nearest grid point to 0.8.")
    args = parser.parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        settings, runs, summary = load_results(args.results_dir)
        sigma_r = args.sigma_r
        sigma_mu = args.sigma_mu
        if sigma_r is None:
            sigma_r = min(settings["grids"]["sigma_r"], key=lambda value: abs(value - 1.0))
        if sigma_mu is None:
            sigma_mu = min(settings["grids"]["sigma_mu"], key=lambda value: abs(value - 0.8))
        mu_heatmap = heatmap_slice(summary, settings, "sigma_mu", "sigma_r", sigma_r)
        r_heatmap = heatmap_slice(summary, settings, "sigma_r", "sigma_mu", sigma_mu)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Loaded {len(runs)} simulations. Selected slices: sigma_r={sigma_r:g}; "
          f"sigma_mu={sigma_mu:g}", flush=True)
    if settings["smoke_test"]:
        print("SMOKE TEST: these figures check execution, not biological behaviour.", flush=True)
    if settings.get("pilot", False):
        print("PILOT: full-size simulations at grid extremes with two seeds.", flush=True)
    fig = plot_heatmaps(mu_heatmap, r_heatmap, sigma_r, sigma_mu,
                        smoke_test=settings["smoke_test"], pilot=settings.get("pilot", False))
    save_figure(fig, args.output_dir, sigma_r, sigma_mu)
    plt.close(fig)


if __name__ == "__main__":
    main()
