#!/usr/bin/env bash
set -euo pipefail
usage() { echo "Usage: bash run_sweep.sh {pbs|local} [--smoke-test]" >&2; exit 2; }
case ${1:-} in pbs|local) mode=$1; shift ;; *) usage ;; esac
if (( $# > 1 )) || [[ $# == 1 && $1 != --smoke-test ]]; then usage; fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python=${DISPERSION_PYTHON:-"$PWD/.venv/bin/python"}
[[ $python == /* && -x $python ]] || { echo "Use .venv or set DISPERSION_PYTHON to an absolute Python path." >&2; exit 1; }
if [[ $mode == pbs ]]; then
    export PATH="$PATH:/opt/pbs/bin"
    command -v qsub >/dev/null || { echo "PBS qsub is unavailable on this host." >&2; exit 1; }
fi
mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
echo "Run directory: $run_dir"
mkdir -p "$run_dir/src/mantishrimp"
cp src/mantishrimp/*.py "$run_dir/src/mantishrimp/"
cp -R reference "$run_dir/"
cp assay.py sweep.py plot_sweep.py settings.json job.sh requirements.txt LICENSE "$run_dir/"
cd -- "$run_dir"
printf '%s\n' "$python" > interpreter.path
tasks=$("$python" -B sweep.py plan "$@")
"$python" -VV > environment.txt
"$python" -m pip freeze >> environment.txt
echo "$tasks new simulations; see plan.csv and reference_used.csv."
if [[ $mode == local ]]; then
    bash job.sh all | tee run.log
else
    array_id=$(qsub -J "1-$tasks" job.sh)
    echo "PBS simulation array: $array_id"
    qsub -N disp_plot -W "depend=afterok:$array_id" job.sh
fi
