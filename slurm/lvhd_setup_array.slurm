#!/bin/bash
#SBATCH --job-name=lvhd-setup
#SBATCH --account=vxk@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Array variant of slurm/lvhd_setup.slurm: one observation per array task.
# Each task runs the Slice -> NSF -> rejection pipeline for a single
# observation independently. Julia DiffEq state isn't shared across tasks;
# each pays ~4-5 min of Julia JIT startup on its own, then runs full speed.
#
# Slice budget per chain is task.num_reference_posterior_samples / num_chains
# so num_chains > 1 doesn't multiply slice work.
#
# Submit:
#   sbatch slurm/lvhd_setup_array.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$WORK/julia-1.10.11/bin:$PATH"
export JULIA_PROJECT="$PWD/.pixi/envs/default/lib/python3.11/site-packages/diffeqtorch/julia"
export JULIA_DEPOT_PATH="$WORK/.julia"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p logs

echo "=========================================================="
echo "Array:    $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "=========================================================="

srun pixi run python scripts/lvhd_setup_one_obs.py \
    --num-observation "$SLURM_ARRAY_TASK_ID" \
    --num-chains 4

echo
echo "=========================================================="
echo "lvhd-setup obs=$SLURM_ARRAY_TASK_ID finished at $(date)"
echo "=========================================================="
