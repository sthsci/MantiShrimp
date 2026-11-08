#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: bash run_sweep.sh {local|pbs|lsf} [--smoke-test]" >&2; exit 2; }
case ${1:-} in local|pbs|lsf) mode=$1; shift ;; *) usage ;; esac
if (( $# > 1 )) || [[ $# == 1 && $1 != --smoke-test ]]; then usage; fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python="$PWD/.venv/bin/python"
[[ -x $python ]] || { echo "Set up .venv on this machine first; see README.md." >&2; exit 1; }

mkdir -p runs
run_dir=$(mktemp -d "$PWD/runs/${mode}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
echo "Run directory: $run_dir"
mkdir -p "$run_dir/src/mantishrimp"
cp src/mantishrimp/*.py "$run_dir/src/mantishrimp/"
cp sweep.py settings.json job.sh requirements.txt LICENSE "$run_dir/"
cd -- "$run_dir"
tasks=$("$python" sweep.py plan "$@")
"$python" -VV > environment.txt
"$python" -m pip freeze >> environment.txt
echo "$tasks replicates; settings and task IDs are saved in plan.csv."

case $mode in
    local)
        bash job.sh all | tee run.log ;;
    pbs)
        array_id=$(qsub -J "1-$tasks" job.sh)
        echo "PBS array: $array_id"
        qsub -N adh_summary -W "depend=afterok:$array_id" job.sh ;;
    lsf)
        name="adh_${run_dir##*/}"
        bsub -J "${name}[1-$tasks]" < job.sh
        bsub -J adh_summary -w "done(\"$name\")" < job.sh ;;
esac
