#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# > 1 )) || [[ $# == 1 && $1 != --smoke-test ]]; then
    echo "Usage: bash run_sweep.sh [--smoke-test]" >&2
    exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python; see README.md for environment setup." >&2
    exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
{
    echo "Run directory: $run_dir"
    mkdir -p "$run_dir/data" "$run_dir/figures"
    cp 10x10.py plot_heatmap.py requirements.txt parameter_test.ipynb test_notebook_settings.py "$run_dir/"
    cp -R src "$run_dir/src"
    .venv/bin/python -m pip freeze > "$run_dir/environment.txt"

    .venv/bin/python "$run_dir/test_notebook_settings.py" -q
    .venv/bin/python -u "$run_dir/10x10.py" "$@" --output-dir "$run_dir/data"
    .venv/bin/python "$run_dir/plot_heatmap.py" \
        "$run_dir/data/paramsweep_kbind_kunbind_10x10_summary.csv" \
        --output-dir "$run_dir/figures"
    echo "Finished. Data, figures and log: $run_dir"
} 2>&1 | tee -a "$run_dir/run.log"
