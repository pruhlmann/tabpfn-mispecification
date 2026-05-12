#!/bin/bash
#SBATCH --job-name=tabpfn-sweep-largecal
#SBATCH --account=vxk@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Run the calibration sweep at the large-ncal point only (default 50000) for
# ONE sbibm task on a single A100. Writes to a separate RESULTS dir so that
# JSON files do not collide with the main sweep at <task>_<misspec>_sweep.json.
#
# Submit:
#   sbatch slurm/sweep_largecal.slurm                  # default task (gaussian_linear)
#   sbatch slurm/sweep_largecal.slurm two_moons        # override task
#
# To merge with the main sweep results for plotting, copy/symlink the JSON
# from results-largecal/ into results/ under a different name.

set -euo pipefail

TASK="${1:-gaussian_linear}"
CONFIG="configs/${TASK}.py"
RESULTS="${RESULTS:-$WORK/projects/tabpfn-mispecification/results-largecal}"
CALIB_SIZES="${CALIB_SIZES:-50000}"
export HF_HOME="$WORK/huggingface_cache"
export TABPFN_MODEL_CACHE_DIR="$HF_HOME"
export TABPFN_DISABLE_TELEMETRY=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Julia 1.10 + patched diffeqtorch deps (see slurm/JEAN_ZAY_SETUP.md)
export PATH="$WORK/julia-1.10.11/bin:$PATH"
export JULIA_PROJECT="$PWD/.pixi/envs/gpu/lib/python3.11/site-packages/diffeqtorch/julia"
export JULIA_DEPOT_PATH="$WORK/.julia"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/gpu/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$HF_HOME" ]]; then
    echo "ERROR: weight cache missing at $HF_HOME." >&2
    echo "GPU nodes have no internet — push weights from a node with internet first:" >&2
    echo "    pixi run sync-tabpfn-weights" >&2
    exit 1
fi

if [[ ! -f "$HF_HOME/tabpfn-v2-regressor.ckpt" ]]; then
    echo "ERROR: tabpfn-v2-regressor.ckpt missing in $HF_HOME." >&2
    echo "Push weights from a node with internet first:" >&2
    echo "    pixi run sync-tabpfn-weights" >&2
    exit 1
fi

mkdir -p logs "$RESULTS"

module purge

echo "=========================================================="
echo "Job:         $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:        $(hostname)"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Task:        $TASK"
echo "Config:      $CONFIG"
echo "Results:     $RESULTS"
echo "Calib sizes: $CALIB_SIZES"
echo "HF_HOME:     $HF_HOME"
echo "=========================================================="

srun pixi run -e gpu sweep --config "$CONFIG" --output_dir "$RESULTS" --calib_sizes "$CALIB_SIZES"

echo
echo "=========================================================="
echo "Largecal sweep for $TASK completed at $(date)"
echo "=========================================================="
