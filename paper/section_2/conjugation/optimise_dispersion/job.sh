#!/usr/bin/env bash
#PBS -N disp_sweep
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=04:00:00
#PBS -j oe
set -euo pipefail
if (( $# == 0 )); then
    cd -- "${PBS_O_WORKDIR:?PBS submission directory missing}"
    if [[ -n ${PBS_ARRAY_INDEX:-} ]]; then set -- task "$PBS_ARRAY_INDEX"; else set -- collect; fi
else
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mplconfig"
IFS= read -r python < interpreter.path
exec "$python" -u sweep.py "$@"
