import sys
import numpy as np
import pandas as pd
import random

if __package__:
    from .simulator import KTSimulator
else:
    from simulator import KTSimulator


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

seed = 42
np.random.seed(seed)
random.seed(seed)


# ─────────────────────────────────────────────────────────────
# Sweep setup
# ─────────────────────────────────────────────────────────────

N_TOTAL = 200

ratio_values = [
    0.05,   # 1:20
    0.1,    # 1:10
    0.2,    # 1:5
    0.5,    # 1:2
    1.0,    # 1:1
    2.0,    # 2:1
    5.0,    # 5:1
    10.0,   # 10:1
]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results = []
killer_results = []
spatial_results = []


# ─────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────

for ratio in ratio_values:
    for run_seed in seeds:

        N_KILLER = int(round(N_TOTAL * ratio / (1 + ratio)))
        N_TARGET = int(N_TOTAL - N_KILLER)

        print(
            f"Running E:T ratio={ratio}, "
            f"N_TARGET={N_TARGET}, N_KILLER={N_KILLER}, "
            f"seed={run_seed}",
            flush=True
        )

        cell_history_df, *_ = KTSimulator(
            N_TARGET=N_TARGET,
            N_KILLER=N_KILLER,
            KREP_KT=250,
            F_A_KT=25,
            KREP_TT=250,
            F_A_TT=1,
            KREP_KK=250,
            F_A_KK=1,
            DT0=0.2,
            DS=1,
            KILLER_MOTILITY=150,
            TARGET_MOTILITY=50,
            SIM_DURATION=25,
            K_BIND=3.0,
            K_UNBIND=0.5,
            RNG_SEED=run_seed,
            ini_seed=42,
        )

        # ─────────────────────────────────────────────────────
        # Final target survival / killing
        # ─────────────────────────────────────────────────────

        target_df = cell_history_df[
            cell_history_df["cell_type"] == "target"
        ].copy()

        killer_df = cell_history_df[
            cell_history_df["cell_type"] == "killer"
        ].copy()

        final_step = target_df["step"].max()
        final_targets = target_df[target_df["step"] == final_step].copy()

        total_targets = len(final_targets)
        alive_targets = int(final_targets["alive_status"].sum())
        kills = total_targets - alive_targets

        killed_targets = final_targets[
            final_targets["alive_status"] == 0
        ].copy()

        # ─────────────────────────────────────────────────────
        # Per-killer kill counts
        # ─────────────────────────────────────────────────────

        killer_ids = (
            killer_df["cell_id"]
            .dropna()
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .astype(int)
            .unique()
        )

        killer_ids = sorted(killer_ids)

        if "killed_by" in killed_targets.columns and len(killed_targets) > 0:

            killed_targets["killed_by_clean"] = killed_targets["killed_by"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )

            killed_targets = killed_targets[
                killed_targets["killed_by_clean"].notna()
            ].copy()

            if len(killed_targets) > 0:
                killed_targets["killed_by_clean"] = (
                    killed_targets["killed_by_clean"].astype(int)
                )
                kill_counts = killed_targets["killed_by_clean"].value_counts()
            else:
                kill_counts = pd.Series(dtype=int)

        else:
            kill_counts = pd.Series(dtype=int)

        obs_time = float(cell_history_df["time"].max())

        for killer_id in killer_ids:
            killer_results.append({
                "E:T ratio": ratio,
                "Killer cell number": N_KILLER,
                "Target cell number": N_TARGET,
                "seed": run_seed,
                "killer_id": int(killer_id),
                "kills_per_killer": int(kill_counts.get(int(killer_id), 0)),
                "obs_time": obs_time,
            })

        # ─────────────────────────────────────────────────────
        # Kill locations
        # For each killed target, record first timestep where alive_status = 0
        # ─────────────────────────────────────────────────────

        if len(killed_targets) > 0:

            killed_ids = killed_targets["cell_id"].unique()

            cols_to_keep = ["cell_id", "step", "x", "y"]
            if "time" in target_df.columns:
                cols_to_keep.append("time")

            kill_steps = (
                target_df[
                    (target_df["cell_id"].isin(killed_ids)) &
                    (target_df["alive_status"] == 0)
                ]
                .sort_values("step")
                .groupby("cell_id", as_index=False)
                .first()[cols_to_keep]
            )

            for _, row in kill_steps.iterrows():

                if "time" in kill_steps.columns:
                    kill_time = float(row["time"])
                else:
                    kill_time = float(row["step"]) * 0.2

                spatial_results.append({
                    "E:T ratio": ratio,
                    "Killer cell number": N_KILLER,
                    "Target cell number": N_TARGET,
                    "seed": run_seed,
                    "cell_id": int(row["cell_id"]),
                    "event": "kill",
                    "step": int(row["step"]),
                    "time": kill_time,
                    "x": row["x"],
                    "y": row["y"],
                    "snap_step": np.nan,
                    "snap_frac": np.nan,
                    "cell_type": "target",
                    "alive_status": 0,
                })

        # ─────────────────────────────────────────────────────
        # Spatial snapshots at 0%, 25%, 50%, 75%, 100%
        # ─────────────────────────────────────────────────────

        snap_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
        available_steps = np.sort(target_df["step"].unique())

        for snap_frac in snap_fracs:

            target_snap_step = int(snap_frac * final_step)

            nearest = available_steps[
                np.argmin(np.abs(available_steps - target_snap_step))
            ]

            snap_cols = ["cell_type", "cell_id", "x", "y", "alive_status"]

            if "time" in cell_history_df.columns:
                snap_cols.append("time")

            snap = cell_history_df[
                cell_history_df["step"] == nearest
            ][snap_cols].copy()

            snap["E:T ratio"] = ratio
            snap["Killer cell number"] = N_KILLER
            snap["Target cell number"] = N_TARGET
            snap["seed"] = run_seed
            snap["event"] = "snapshot"
            snap["step"] = nearest
            snap["snap_step"] = nearest
            snap["snap_frac"] = snap_frac

            if "time" not in snap.columns:
                snap["time"] = float(nearest) * 0.2

            spatial_results.extend(snap.to_dict("records"))

        # ─────────────────────────────────────────────────────
        # Contacts and synapses
        # ─────────────────────────────────────────────────────

        all_contacts = target_df["contacts"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).sum()

        target_df = target_df.sort_values(["cell_id", "step"])

        synapses = 0

        for _, g in target_df.groupby("cell_id"):

            prev = set()

            for contacts in g["contacts"]:

                curr = set(contacts) if isinstance(contacts, list) else set()

                # New synapse/contact event = contact appears now but was absent previously
                synapses += len(curr - prev)

                prev = curr

        # ─────────────────────────────────────────────────────
        # Summary row
        # ─────────────────────────────────────────────────────

        results.append({
            "E:T ratio": ratio,
            "Killer cell number": N_KILLER,
            "Target cell number": N_TARGET,
            "seed": run_seed,
            "kills": int(kills),
            "contacts": int(all_contacts),
            "synapses": int(synapses),
            "killing_efficiency": kills / total_targets if total_targets > 0 else np.nan,
            "kills_per_synapse": kills / synapses if synapses > 0 else np.nan,
            "contacts_per_target": all_contacts / total_targets if total_targets > 0 else np.nan,
            "obs_time": obs_time,
        })


# ─────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
killer_results_df = pd.DataFrame(killer_results)
spatial_df = pd.DataFrame(spatial_results)

results_df.to_csv(
    "paramsweep_ET_ratio_summary.csv.gz",
    index=False,
    compression="gzip"
)

killer_results_df.to_csv(
    "paramsweep_ET_ratio_killerresults.csv.gz",
    index=False,
    compression="gzip"
)

spatial_df.to_csv(
    "paramsweep_ET_ratio_spatial.csv.gz",
    index=False,
    compression="gzip"
)

print("Done.")
print(f"Saved summary rows: {len(results_df)}")
print(f"Saved killer rows: {len(killer_results_df)}")
print(f"Saved spatial rows: {len(spatial_df)}")