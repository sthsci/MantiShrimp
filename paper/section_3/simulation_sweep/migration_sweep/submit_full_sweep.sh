#!/usr/bin/env bash
# Run on the Imperial login node, beside run_sweep.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
BATCH="${EPHEMERAL:-/rds/general/user/$(id -un)/ephemeral}/section3_grid10"

command -v qsub >/dev/null || { echo "Run this on the Imperial login node." >&2; exit 1; }
if [[ -e "$BATCH" ]]; then
    echo "$BATCH already exists. No new jobs submitted. Check qstat -u \$USER." >&2
    exit 1
fi
bash run_sweep.sh --dry-run --workers 8
mkdir -- "$BATCH"

.venv/bin/python - <<'PY' > "$BATCH/slices.tsv"
import itertools, json
with open("section2_grid.json") as f:
    grid = json.load(f)
if any(len(grid[k]) != 10 or len(set(grid[k])) != 10 for k in ("alpha", "sigma_mu", "sigma_r")):
    raise SystemExit("Expected 10 distinct values on each of the three axes.")
for i, (alpha, mu) in enumerate(itertools.product(grid["alpha"], grid["sigma_mu"]), 1):
    print(f"{i:03d}", alpha, mu, sep="\t")
PY

echo "Writing simulation results to $BATCH"
while read -r id alpha mu; do
    if ! reply=$(python ~/bin/bsub -n "s3g_$id" -p 8 -m 8 -t 08:00:00 \
        env "SWEEP_RUNS_DIR=$BATCH/$id" bash "$PWD/run_sweep.sh" \
        --alpha "$alpha" --sigma-mu "$mu" --cells 1000 --record-every 1 --workers 8 </dev/null); then
        printf '%s\nbsub failed for slice %s. Check qstat before retrying.\n' "$reply" "$id" >&2
        exit 1
    fi
    # bsub can return success even when qsub rejects a job.
    job=$(awk '/^[[:space:]]*[0-9]+([.][[:alnum:]_.-]+)?[[:space:]]*$/ {print $1; exit}' <<< "$reply")
    if [[ -z "$job" ]]; then
        printf '%s\nNo PBS job ID for slice %s. Stopping; check qstat before retrying.\n' "$reply" "$id" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\n' "$id" "$alpha" "$mu" "$job" >> "$BATCH/submitted.tsv"
    echo "[$id/100] submitted $job"
done < "$BATCH/slices.tsv"
