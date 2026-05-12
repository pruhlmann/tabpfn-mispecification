#!/bin/bash
#SBATCH --job-name=lvhd-diagnose
#SBATCH --account=vxk@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Short diagnostic for the LV-HD reference-posterior pipeline. Calls the
# slice+NSF+rejection helper once with tiny budgets (num_samples=200,
# num_warmup=200, batch_size=2000, num_batches_without_new_max=20) on
# observation 1, then prints per-stage walltime + log_prob call/failure
# counts. Used to decide where to spend the 20h walltime budget in
# slurm/lvhd_setup.slurm.
#
# Does NOT write to tabpfn_misspec/lotka_volterra_hd/files/ (the helper is
# called directly, not via task._setup()).
#
# Submit:
#   sbatch slurm/lvhd_diagnose.slurm

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
echo "Job:      $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "=========================================================="

srun pixi run python scripts/lvhd_diagnose.py

echo
echo "=========================================================="
echo "Diagnostic completed at $(date)"
echo "=========================================================="
