#!/bin/bash
#SBATCH --job-name=lvhd-setup
#SBATCH --account=vxk@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generate reference posterior + true parameters + observations for
# lotka_volterra_hd. Long-running CPU job (sbibm Appendix B.1 pipeline:
# Slice-MCMC via sbi.samplers.mcmc.slice_numpy.SliceSamplerVectorized -> NSF
# density estimator -> rejection sampling). Outputs land under
# tabpfn_misspec/lotka_volterra_hd/files/num_observation_{1,2,3}/.
#
# n_jobs=1: observations run sequentially. Julia DiffEq state isn't safe to
# clone across joblib loky workers; parallelizing here is a future SLURM array
# job, not a per-process concern.
#
# Submit:
#   sbatch slurm/lvhd_setup.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Julia 1.10 + patched diffeqtorch deps (see slurm/JEAN_ZAY_SETUP.md).
# CPU env: setup uses pyro/sbi without GPU torch.
export PATH="$WORK/julia-1.10.11/bin:$PATH"
export JULIA_PROJECT="$PWD/.pixi/envs/default/lib/python3.11/site-packages/diffeqtorch/julia"
export JULIA_DEPOT_PATH="$WORK/.julia"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"

# Let numpy/torch/MKL spread NSF training and rejection-sampling batched
# log_prob evals across the requested CPUs. Without these, BLAS often defaults
# to 1 thread under SLURM cgroups.
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

srun pixi run python -c "from tabpfn_misspec.tasks import get_task; \
    t = get_task('lotka_volterra_hd'); t._setup(n_jobs=1)"

echo
echo "=========================================================="
echo "Setup for lotka_volterra_hd completed at $(date)"
echo "=========================================================="
