#!/bin/bash
#OAR -n tabpfn_sweep
#OAR -l /nodes=1/gpu=1,walltime=12:00:00
#OAR -t besteffort
#OAR -t idempotent
#OAR --array 4

# Array job script - one sweep per task config
# Submit with: oarsub -S ./scripts/cluster/run_sweep_array.sh

set -e

PROJECT_DIR="/home/pruhlman/project/tabpfn-misspecification"
RESULTS_DIR="/scratch/clear/pruhlman/tabpfn-misspecification/results"

export HF_HOME="/scratch/clear/pruhlman/huggingface_cache"
mkdir -p "$HF_HOME"

# Map array index to task config (OAR arrays are 1-indexed)
CONFIGS=("configs/gaussian_linear.py" "configs/two_moons.py" "configs/gaussian_mixture.py" "configs/slcp.py")
CONFIG_IDX=$((OAR_ARRAY_INDEX - 1))
CONFIG="${CONFIGS[$CONFIG_IDX]}"

cd "$PROJECT_DIR"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "TabPFN Misspecification Sweep"
echo "Config: $CONFIG"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Array Index: $OAR_ARRAY_INDEX"
echo "============================================================"

pixi run sweep -- --config "$CONFIG" --output_dir "$RESULTS_DIR"

echo "Sweep for $CONFIG completed at $(date)"
