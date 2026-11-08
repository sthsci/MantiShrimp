#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=${SWEEP_PYTHON:-"$bundle/.venv/bin/python"}
if [[ ! -x "$python" ]]; then
    echo "Missing $python. Create the cluster .venv as described in README.md." >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

source_dir=$bundle
run_dir=
if [[ ${1:-} == --resume ]]; then
    [[ -n ${2:-} ]] || { echo "Usage: bash run_sweep.sh --resume RUN_DIRECTORY [--workers N]" >&2; exit 2; }
    run_dir=$(cd -- "$2" && pwd -P)
    shift 2
    source_dir="$run_dir/source"
fi
preview=false
for argument in "$@"; do
    case "$argument" in
        --output-dir|--output-dir=*|--resume|--resume=*)
            echo "Set SWEEP_RUNS_DIR for new jobs; for a restart put --resume RUN_DIRECTORY first." >&2
            exit 2 ;;
        --dry-run|--help|-h) preview=true ;;
    esac
done
if [[ -n $run_dir ]]; then set -- --resume "$run_dir/data" "$@"; fi
if [[ $preview == true ]]; then exec "$python" "$source_dir/run_sweep.py" "$@"; fi

if [[ -z $run_dir ]]; then
    "$python" "$bundle/run_sweep.py" --dry-run "$@" >/dev/null
    runs_root=${SWEEP_RUNS_DIR:-"$bundle/runs"}
    mkdir -p -- "$runs_root"
    runs_root=$(cd -- "$runs_root" && pwd -P)
    run_dir=$(mktemp -d "$runs_root/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    source_dir="$run_dir/source"
    mkdir -p -- "$source_dir/src/mantishrimp"
    cp -- "$bundle"/{run_sweep.py,timeline.py,test_run_sweep.py,submit_full_sweep.py,submit_full_sweep.sh,run_sweep.sh,section2_grid.json,requirements.txt,README.md,LICENSE,SHA256SUMS} "$source_dir/"
    cp -- "$bundle"/src/mantishrimp/*.py "$source_dir/src/mantishrimp/"
    "$python" -m pip freeze > "$run_dir/environment.txt"
    set -- --output-dir "$run_dir/data" "$@"
fi
{
    echo "Run directory: $run_dir"
    echo "Started: $(date -u +%FT%TZ) on $(hostname); scheduler job: ${PBS_JOBID:-${LSB_JOBID:-local}}"
    "$python" -V
    printf 'Arguments:'
    printf ' %q' "$@"
    printf '\n'
    "$python" -u "$source_dir/run_sweep.py" "$@"
    echo "Finished: $run_dir/data/summary.csv"
} 2>&1 | tee -a "$run_dir/run.log"
