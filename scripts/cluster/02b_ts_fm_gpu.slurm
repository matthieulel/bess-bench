#!/bin/bash
#SBATCH --job-name=bess-02b-tsfm
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/02b_tsfm_%j.out
#SBATCH --error=logs/02b_tsfm_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 2b — EWForecast: time-series foundation models (GPU)
# ════════════════════════════════════════════════════════════════════════════
#
# Zero-shot Chronos-Bolt (small, base) and TimesFM-2.0-500M, then the
# additional forecasters whose packages are installed: trained linear head on
# frozen Chronos-Bolt, Time-MoE, Moirai, Moirai-MoE (see requirements.txt).
#
# Requires Phase 0 (seed-42 embeddings) and Phase 1 (features).
# Outputs : reference_results/downstream/t3_*.json
# Duration : ~15–30 min on one GPU, < 1 h with the additional forecasters.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 2b — EWForecast time-series foundation models"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

cd "$REPO_ROOT"
OUT="$REF/downstream"
EMB42="$(emb_dir 42)/star_data.pt"

python downstream/t3_ts_fm_baselines.py \
    --embeddings "$EMB42" \
    --features   "$SPEC" \
    --output_dir "$OUT" \
    --device cuda \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group t3_ts_fm

export BESS_EMB_PATH="$EMB42" BESS_FEATURES_PATH="$SPEC" BESS_DOWNSTREAM_DIR="$OUT"
run_if_importable() {   # <python module> <script>
    if python -c "import $1" 2>/dev/null; then
        echo ""
        echo ">>> $2  $(date)"
        python "downstream/$2"
    else
        echo ">>> SKIP $2 (python package '$1' not installed)"
    fi
}
run_if_importable chronos      t3_chronos_zeroshot_samepairs.py
run_if_importable chronos      t3_chronos_trained_head.py
run_if_importable transformers t3_timemoe_zeroshot.py
run_if_importable uni2ts       t3_moirai_zeroshot.py
run_if_importable uni2ts       t3_moiraimoe_zeroshot.py

echo ""
echo "PHASE 2b done  —  $(date)"
