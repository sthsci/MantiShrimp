#!/usr/bin/env bash
#PBS -N adhesion
#PBS -S /bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=04:00:00
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 4:00
#BSUB -o adhesion.%J.%I.log
set -euo pipefail

if (( $# == 0 )); then
    # Schedulers may run a spooled copy of this file outside the submission directory.
    cd -- "${PBS_O_WORKDIR:-${LS_SUBCWD:-$PWD}}"
    index=${PBS_ARRAY_INDEX:-${LSB_JOBINDEX:-0}}
    if [[ $index == 0 ]]; then set -- collect; else set -- task "$index"; fi
else
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.mplconfig"
exec ../../.venv/bin/python -u sweep.py "$@"
