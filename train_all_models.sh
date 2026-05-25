#!/bin/bash
#SBATCH --job-name=train_models_array
#SBATCH --partition=main
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-149%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs results models

module load miniconda3
source activate ml_models

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PARQUET_LIST="parquet_files.txt"

if [[ ! -f "$PARQUET_LIST" ]]; then
    echo "Missing $PARQUET_LIST"
    exit 1
fi

N_FILES=$(wc -l < "$PARQUET_LIST")
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [[ "$TASK_ID" -ge "$N_FILES" ]]; then
    echo "Task ID $TASK_ID is out of range for $N_FILES files, exiting."
    exit 0
fi

PARQUET_PATH=$(sed -n "$((TASK_ID + 1))p" "$PARQUET_LIST")

echo "======================================"
echo "Job started on: $(hostname)"
echo "Task ID:        ${SLURM_ARRAY_TASK_ID}"
echo "Parquet path:   ${PARQUET_PATH}"
echo "Working dir:    $(pwd)"
echo "CPUs:           ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory:         ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "======================================"

python -u train_all_models.py "$PARQUET_PATH"

echo "Finished task ${SLURM_ARRAY_TASK_ID}"
