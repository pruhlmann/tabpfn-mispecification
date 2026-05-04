#!/bin/bash
#SBATCH --job-name=tabpfn-sweep
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

# Run the calibration sweep for ONE sbibm task on a single A100.
#
# Submit:
#   sbatch slurm/sweep.slurm                  # default task (gaussian_linear)
#   sbatch slurm/sweep.slurm two_moons        # override task
#
# Available tasks: any configs/<name>.py (gaussian_linear, two_moons,
# gaussian_mixture, slcp, sir, lotka_volterra, gaussian_linear_hd).

set -euo pipefail

TASK="${1:-gaussian_linear}"
CONFIG="configs/${TASK}.py"
RESULTS="${RESULTS:-$WORK/projects/tabpfn-mispecification/results}"
export HF_HOME="$WORK/huggingface_cache"
export TABPFN_MODEL_CACHE_DIR="$HF_HOME"
export TABPFN_DISABLE_TELEMETRY=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

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
echo "Job:      $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:     $(hostname)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Task:     $TASK"
echo "Config:   $CONFIG"
echo "Results:  $RESULTS"
echo "HF_HOME:  $HF_HOME"
echo "=========================================================="

srun pixi run -e gpu sweep --config "$CONFIG" --output_dir "$RESULTS"

echo
echo "=========================================================="
echo "Sweep for $TASK completed at $(date)"
echo "=========================================================="
