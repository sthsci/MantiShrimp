#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python; follow the environment setup in README.md." >&2
    exit 1
fi

for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        exec .venv/bin/python motility_sweep.py "$@"
    fi
done

# Each process uses one numerical-library thread, matching --workers to CPU cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export MPLBACKEND=Agg MPLCONFIGDIR="$run_dir/.mplconfig"
{
    echo "Run directory: $run_dir"
    cp ./*.py run_sweep.sh requirements.txt README.md LICENSE "$run_dir/"
    mkdir -p "$run_dir/src/mantishrimp"
    cp src/mantishrimp/*.py "$run_dir/src/mantishrimp/"
    .venv/bin/python -m pip freeze > "$run_dir/environment.txt"
    .venv/bin/python -V
    printf 'Arguments:'
    printf ' %q' "$@"
    printf '\n'

    .venv/bin/python -u "$run_dir/motility_sweep.py" --output-dir "$run_dir/data" "$@"
    .venv/bin/python "$run_dir/plot_heatmaps.py" "$run_dir/data" --output-dir "$run_dir/figures"
    echo "Finished. Data, figures and log: $run_dir"
} 2>&1 | tee -a "$run_dir/run.log"
