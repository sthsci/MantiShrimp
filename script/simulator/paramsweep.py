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

k_bind_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
seeds = [1, 2, 3, 4, 5]

results = []
killer_results = []

for k in k_bind_values:
    for run_seed in seeds:
        print(f"Running K_BIND={k}, seed={run_seed}", flush=True)

        cell_history_df, *_ = KTSimulator(
            # fixed params
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
            SIM_DURATION=10,

            # key variables
            K_BIND=k,
            K_UNBIND=0.5,
            RNG_SEED=run_seed,
            ini_seed=42,
        )

        target_df = cell_history_df[cell_history_df['cell_type'] == 'target'].copy()
        final_step = target_df['step'].max()
        final_targets = target_df[target_df['step'] == final_step]

        total_targets = len(final_targets)
        alive_targets = int(final_targets['alive_status'].sum())
        kills = total_targets - alive_targets
        killed_targets = final_targets[final_targets['alive_status'] == 0]

        killer_df = cell_history_df[cell_history_df["cell_type"] == "killer"].copy()

        # Get one clean scalar ID per killer
        killer_ids = (
            killer_df["cell_id"]
            .dropna()
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .astype(int)
            .unique()
        )

        killer_ids = sorted(killer_ids)

        # Clean killed_by column
        killed_targets = killed_targets.copy()

        killed_targets["killed_by_clean"] = killed_targets["killed_by"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

        killed_targets = killed_targets[killed_targets["killed_by_clean"].notna()]
        killed_targets["killed_by_clean"] = killed_targets["killed_by_clean"].astype(int)

        # Count how many targets each killer killed
        kill_counts = killed_targets["killed_by_clean"].value_counts()

        for killer_id in killer_ids:
            killer_results.append({
                "k_bind": k,
                "seed": run_seed,
                "killer_id": int(killer_id),
                "kills_per_killer": int(kill_counts.get(int(killer_id), 0)),
                "obs_time": float(cell_history_df["time"].max()),
            })

        # (A) All contacts/contact-time: sum of bound partners over all target-steps
        all_contacts = target_df['contacts'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).sum()

        # (B) True binding events/new synapses: newly appearing killer IDs per target across steps
        target_df = target_df.sort_values(['cell_id', 'step'])
        synapses = 0

        for _, g in target_df.groupby('cell_id'):
            prev = set()

            for contacts in g['contacts']:
                curr = set(contacts) if isinstance(contacts, list) else set()
                synapses += len(curr - prev)
                prev = curr

        results.append({
            'k_bind': k,
            'seed': run_seed,
            'kills': kills,
            'contacts': all_contacts,
            'synapses': synapses,
        })

results_df = pd.DataFrame(results)
killer_results_df = pd.DataFrame(killer_results)

results_df.to_csv("outputs/paramsweep_HPC.csv.gz", index=False, compression='gzip')
killer_results_df.to_csv("outputs/paramsweep_killerresults.csv.gz", index=False, compression='gzip')
