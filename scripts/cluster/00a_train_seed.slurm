#!/bin/bash
#SBATCH --job-name=bess-00a-train
#SBATCH --partition=shortrun,longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/00a_train_%j.out
#SBATCH --error=logs/00a_train_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 0a — Train a SINGLE BeMAE-Hα seed (parallelisable)
# ════════════════════════════════════════════════════════════════════════════
#
# Required env var :  SEED  (42, 123 or 456)
#
# Usage :
#   sbatch --export=ALL,SEED=42 00a_train_seed.slurm
#
# Produces : stage1_encoder/runs/Halpha_all_seed${SEED}/best.pt
#            stage1_encoder/runs/Halpha_all_seed${SEED}/config.json
#
# Duration : 41 to 51 min on one RTX 2080 Ti
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

if [[ -z "${SEED:-}" ]]; then
    echo "ERROR: SEED env var must be set (e.g. --export=ALL,SEED=42)"
    exit 1
fi

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 0a — BeMAE-Hα train (seed=${SEED})"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

cd "$REPO_ROOT/stage1_encoder"
mkdir -p runs

RUN_NAME="Halpha_all_seed${SEED}"
RUN_DIR="runs/${RUN_NAME}"

if [[ -f "${RUN_DIR}/best.pt" ]]; then
    echo ">>> SKIP train (${RUN_DIR}/best.pt already exists)"
    exit 0
fi

echo ">>> TRAIN seed=${SEED}  $(date)"
python train.py \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --output_dir runs \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group bemae_pretrain_v1 \
    --wandb_tags "multiseed"

echo ">>> Done seed=${SEED}  $(date)"
