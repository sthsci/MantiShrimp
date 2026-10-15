import sys
import numpy as np
import pandas as pd
import random

if __package__:
    from .simulator import KTSimulator
else:
    from simulator import KTSimulator

seed = 42
np.random.seed(seed)
random.seed(seed)

# ─── Sweep parameters ─────────────────────────────────────────────────────────

F_A_KT_values = [1, 5, 10, 25, 50, 100, 150, 200]
seeds         = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Timepoints (hours) at which to snapshot the simulation state
SNAPSHOT_TIMES = [0, 5, 10, 15, 20, 25]

# ─── Exhaustion classifier ────────────────────────────────────────────────────

def is_exhausted(seq: np.ndarray) -> bool:
    """Starts killing (1s), then permanently stops (trailing 0s): 111...000"""
    if seq.size == 0:
        return False
    if not (seq[0] == 1 and seq[-1] == 0):
        return False
    first_zero = np.where(seq == 0)[0]
    if first_zero.size == 0:
        return False
    z0 = first_zero[0]
    return np.all(seq[z0:] == 0)


def classify_killer(seq: np.ndarray) -> str:
    seq = np.asarray(seq, dtype=int)
    if seq.size == 0:
        return "no_interaction"
    elif np.all(seq == 0):
        return "no_kill"
    elif np.all(seq == 1):
        return "serial_kill"
    elif is_exhausted(seq):
        return "exhausted"
    else:
        return "stochastic"


def build_kill_sequence(killer_id: int, cell_history_df: pd.DataFrame) -> np.ndarray:
    target_df = (
        cell_history_df[cell_history_df["cell_type"] == "target"]
        .sort_values(["cell_id", "step"])
        .copy()
    )

    def _parse(val):
        if isinstance(val, list):
            return val
        try:
            import ast
            return ast.literal_eval(str(val))
        except Exception:
            return []

    target_df["contacts_parsed"] = target_df["contacts"].apply(_parse)

    def _clean_killed_by(val):
        if isinstance(val, list):
            return val[0] if len(val) > 0 else None
        return val

    target_df["killed_by_clean"] = target_df["killed_by"].apply(_clean_killed_by)

    episodes = []

    for target_id, g in target_df.groupby("cell_id", sort=False):
        g = g.sort_values("step")
        steps    = g["step"].to_numpy()
        contacts = g["contacts_parsed"].to_numpy()
        kb_vals  = g["killed_by_clean"].to_numpy()

        kill_step = None
        for step, kb in zip(steps, kb_vals):
            try:
                if int(kb) == killer_id:
                    kill_step = step
                    break
            except (TypeError, ValueError):
                pass

        in_contact    = False
        contact_start = None

        for i, (step, curr_contacts) in enumerate(zip(steps, contacts)):
            was_in = killer_id in [int(c) for c in curr_contacts if c is not None]

            if was_in and not in_contact:
                in_contact    = True
                contact_start = step

            elif not was_in and in_contact:
                if kill_step is not None and contact_start <= kill_step < step:
                    episodes.append(1)
                    kill_step = None
                else:
                    episodes.append(0)
                in_contact    = False
                contact_start = None

        if in_contact:
            if kill_step is not None and contact_start <= kill_step <= steps[-1]:
                episodes.append(1)
            else:
                episodes.append(0)

    return np.array(episodes, dtype=int)


# ─── Helper: nearest step to a target time ────────────────────────────────────

def _nearest_step(df: pd.DataFrame, t: float) -> int:
    """Return the step whose recorded time is closest to t."""
    times = df.drop_duplicates("step")[["step", "time"]].set_index("step")["time"]
    return int((times - t).abs().idxmin())


# ─── Results containers ───────────────────────────────────────────────────────

results           = []   # per-run summary (unchanged)
killer_results    = []   # per-killer classification (unchanged)
timepoint_results = []   # NEW: per-run × timepoint snapshots


# ─── Main sweep ───────────────────────────────────────────────────────────────

for f_a_kt in F_A_KT_values:
    for run_seed in seeds:
        print(f"Running F_A_KT={f_a_kt}, seed={run_seed}", flush=True)

        cell_history_df, *_ = KTSimulator(
            KREP_KT=250,
            F_A_KT=f_a_kt,
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

        # ── Separate killer / target dataframes ───────────────────────────────

        killer_df = cell_history_df[cell_history_df["cell_type"] == "killer"].copy()
        target_df = cell_history_df[cell_history_df["cell_type"] == "target"].copy()

        # ── End-of-simulation kill/target summary ─────────────────────────────

        final_step    = target_df["step"].max()
        final_targets = target_df[target_df["step"] == final_step]

        total_targets = len(final_targets)
        alive_targets = int(final_targets["alive_status"].sum())
        kills         = total_targets - alive_targets

        killed_targets = final_targets[final_targets["alive_status"] == 0].copy()
        killed_targets["killed_by_clean"] = killed_targets["killed_by"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        killed_targets = killed_targets[killed_targets["killed_by_clean"].notna()]
        killed_targets["killed_by_clean"] = killed_targets["killed_by_clean"].astype(int)
        kill_counts = killed_targets["killed_by_clean"].value_counts()

        # ── Killer IDs ────────────────────────────────────────────────────────

        killer_ids = (
            killer_df["cell_id"]
            .dropna()
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .astype(int)
            .unique()
        )
        killer_ids = sorted(killer_ids)

        # ── Per-killer classification ──────────────────────────────────────────

        state_counts = {
            "no_interaction": 0,
            "no_kill":        0,
            "exhausted":      0,
            "stochastic":     0,
            "serial_kill":    0,
        }

        # Cytotoxic particles released per NK cell:
        # cell_state starts at 1.0 and decays as the NK cell kills.
        # Total drop in cell_state = proxy for cumulative granule secretion.
        # We record this here for the per-killer table.
        killer_state_initial = (
            killer_df.sort_values("step")
            .groupby("cell_id")["cell_state"]
            .first()
        )
        killer_state_final = (
            killer_df.sort_values("step")
            .groupby("cell_id")["cell_state"]
            .last()
        )

        for killer_id in killer_ids:
            seq   = build_kill_sequence(killer_id, cell_history_df)
            state = classify_killer(seq)
            state_counts[state] += 1

            # Cytotoxic output: drop in cell_state (granule depletion proxy)
            s0 = killer_state_initial.get(killer_id, 1.0)
            s1 = killer_state_final.get(killer_id, s0)
            cytotoxic_particles_released = float(s0 - s1)   # 0 → 1 scale

            killer_results.append({
                "F_A_KT":                    f_a_kt,
                "seed":                      run_seed,
                "killer_id":                 int(killer_id),
                "kills_per_killer":          int(kill_counts.get(int(killer_id), 0)),
                "n_episodes":                int(len(seq)),
                "n_kill_episodes":           int(seq.sum()) if len(seq) > 0 else 0,
                "state":                     state,
                "obs_time":                  float(cell_history_df["time"].max()),
                # NEW
                "cytotoxic_particles_released": cytotoxic_particles_released,
                "cell_state_initial":        float(s0),
                "cell_state_final":          float(s1),
            })

        # ── Contacts and synapses ──────────────────────────────────────────────

        all_contacts = target_df["contacts"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).sum()

        target_df_sorted = target_df.sort_values(["cell_id", "step"])
        synapses = 0
        for _, g in target_df_sorted.groupby("cell_id"):
            prev = set()
            for contacts in g["contacts"]:
                curr = set(contacts) if isinstance(contacts, list) else set()
                synapses += len(curr - prev)
                prev = curr

        results.append({
            "F_A_KT":            f_a_kt,
            "seed":              run_seed,
            "kills":             kills,
            "total_targets":     total_targets,
            "contacts":          all_contacts,
            "synapses":          synapses,
            "kill_frac":         kills / total_targets if total_targets > 0 else 0,
            "kills_per_synapse": kills / synapses if synapses > 0 else None,
            **{f"n_{k}": v for k, v in state_counts.items()},
            "frac_exhausted":    state_counts["exhausted"] / len(killer_ids) if killer_ids else None,
            "frac_serial":       state_counts["serial_kill"] / len(killer_ids) if killer_ids else None,
            "frac_no_kill":      state_counts["no_kill"] / len(killer_ids) if killer_ids else None,
        })

        # ── NEW: Timepoint snapshots ───────────────────────────────────────────
        # For each snapshot time, query the simulation state at the nearest recorded step.
        # Records:
        #   n_exhausted_t       — number of NK cells whose cell_state has dropped below
        #                         the exhaustion threshold (0.05) by this timepoint
        #   n_kills_t           — cumulative targets dead by this timepoint
        #   mean_killing_rate_t — mean instantaneous kill rate across all active synapses
        #                         at this step, approximated as ΔDeathFactor/Δt per target
        #   total_cytotoxic_t   — sum of cell_state drops across all NK cells up to this
        #                         timepoint (cumulative granule secretion proxy)

        EXHAUSTION_STATE_THRESHOLD = 0.05  # cell_state below this = functionally exhausted

        # Pre-sort for efficiency
        killer_df_sorted = killer_df.sort_values(["cell_id", "step"])
        target_df_sorted = target_df.sort_values(["cell_id", "step"])

        for t_snap in SNAPSHOT_TIMES:
            step_snap = _nearest_step(cell_history_df, t_snap)
            actual_time = cell_history_df[cell_history_df["step"] == step_snap]["time"].iloc[0]

            # ── NK cells exhausted by this timepoint ──────────────────────────
            # An NK cell is "exhausted" at time t if its cell_state ≤ threshold
            # at any step up to step_snap (i.e. it has depleted its granules)
            killer_snap = killer_df_sorted[killer_df_sorted["step"] <= step_snap]
            min_state_per_killer = killer_snap.groupby("cell_id")["cell_state"].min()
            n_exhausted_t = int((min_state_per_killer <= EXHAUSTION_STATE_THRESHOLD).sum())

            # ── Cumulative kills by this timepoint ────────────────────────────
            # A target is dead if alive_status == False at step_snap
            target_snap = target_df_sorted[target_df_sorted["step"] == step_snap]
            n_kills_t   = int((target_snap["alive_status"] == False).sum())

            # ── Cumulative cytotoxic secretion up to this timepoint ───────────
            # Sum of (initial_state - current_state) across all NK cells
            # = total granule content released up to this timepoint
            killer_at_snap = killer_df_sorted[killer_df_sorted["step"] == step_snap]
            killer_state_t = killer_at_snap.set_index("cell_id")["cell_state"]
            cytotoxic_cumul_t = float(
                (killer_state_initial - killer_state_t.reindex(killer_state_initial.index).fillna(killer_state_initial)).sum()
            )

            # ── Instantaneous killing rate at this timepoint ──────────────────
            # Killing rate ≈ ΔDeathFactor / Δt for targets that are still alive
            # We use the two nearest steps around step_snap to estimate the derivative
            steps_available = sorted(target_df_sorted["step"].unique())
            idx_snap = steps_available.index(step_snap) if step_snap in steps_available else -1

            if idx_snap > 0:
                step_prev  = steps_available[idx_snap - 1]
                t_curr     = target_df_sorted[target_df_sorted["step"] == step_snap]["time"].iloc[0]
                t_prev     = target_df_sorted[target_df_sorted["step"] == step_prev]["time"].iloc[0]
                dt_local   = t_curr - t_prev

                df_curr = (
                    target_df_sorted[target_df_sorted["step"] == step_snap]
                    .set_index("cell_id")["Death_Factor"]
                )
                df_prev = (
                    target_df_sorted[target_df_sorted["step"] == step_prev]
                    .set_index("cell_id")["Death_Factor"]
                )
                delta_df = (df_curr - df_prev.reindex(df_curr.index).fillna(0))
                # Only count targets still alive (Death_Factor < threshold) and
                # where rate is positive (active killing happening)
                alive_mask = target_snap.set_index("cell_id")["alive_status"].reindex(df_curr.index).fillna(True)
                active_rate = delta_df[alive_mask & (delta_df > 0)]
                mean_killing_rate_t = float(active_rate.sum() / dt_local) if dt_local > 0 and len(active_rate) > 0 else 0.0
            else:
                mean_killing_rate_t = 0.0

            timepoint_results.append({
                "F_A_KT":                 f_a_kt,
                "seed":                   run_seed,
                "snapshot_time_requested": t_snap,
                "snapshot_time_actual":   float(actual_time),
                "step":                   step_snap,
                # NK exhaustion
                "n_exhausted_t":          n_exhausted_t,
                # Killing
                "n_kills_t":              n_kills_t,
                "mean_killing_rate_t":    mean_killing_rate_t,
                # Cytotoxic output
                "cytotoxic_cumul_t":      cytotoxic_cumul_t,
            })


# ─── Save ─────────────────────────────────────────────────────────────────────

results_df          = pd.DataFrame(results)
killer_results_df   = pd.DataFrame(killer_results)
timepoint_results_df = pd.DataFrame(timepoint_results)

results_df.to_csv(
    "outputs/paramsweep_FAKT_exhaustion_HPC.csv.gz",
    index=False, compression="gzip"
)
killer_results_df.to_csv(
    "outputs/paramsweep_FAKT_exhaustion_killerresults.csv.gz",
    index=False, compression="gzip"
)
timepoint_results_df.to_csv(
    "outputs/paramsweep_FAKT_exhaustion_timepoints.csv.gz",
    index=False, compression="gzip"
)

print("\nDone.")
print(results_df.groupby("F_A_KT")[["kill_frac", "frac_exhausted", "frac_serial"]].mean().round(3))
print("\nTimepoint results preview:")
print(
    timepoint_results_df
    .groupby(["F_A_KT", "snapshot_time_requested"])[
        ["n_exhausted_t", "n_kills_t", "mean_killing_rate_t", "cytotoxic_cumul_t"]
    ]
    .mean()
    .round(3)
)