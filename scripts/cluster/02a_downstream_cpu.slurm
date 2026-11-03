#!/bin/bash
#SBATCH --job-name=bess-02a-downstream
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/02a_downstream_%j.out
#SBATCH --error=logs/02a_downstream_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 2a — CPU downstream tasks
# ════════════════════════════════════════════════════════════════════════════
#
# Runs every downstream evaluation that does NOT need a GPU :
#
#   SpecProbe     probe_features.py           × 3 encoder seeds (star-grouped CV)
#   SNR bins      snr_ablation.py             × 3 encoder seeds
#   LineTransfer  t2_cross_line_probing.py    (Hβ -> Hα, star-grouped CV)
#   EWForecast    t3_pca_ridge_temporal_baseline.py, t3_z_ridge_temporal_baseline.py,
#                 t3_time_aware_ridge.py, t3_pca_ridge_select_config.py
#   Diagnostics   compute_sigma_ew_and_scree.py
#
# Requires Phase 0 (embeddings) and Phase 1 (features).
# Outputs : reference_results/downstream/*.json
# Duration : ~30 min – 1 h on 8 CPU cores.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 2a — CPU downstream tasks"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

cd "$REPO_ROOT"
OUT="$REF/downstream"
EMB42="$(emb_dir 42)/star_data.pt"

# ── SpecProbe and SNR diagnostic × 3 encoder seeds ──────────────────────────
for ENC in "${SEEDS[@]}"; do
    EMB="$(emb_dir "$ENC")/star_data.pt"
    echo ""
    echo ">>> SpecProbe, encoder seed=${ENC}  $(date)"
    python downstream/probe_features.py \
        --embeddings "$EMB" \
        --features   "$SPEC" \
        --seed "$PROBE_SEED" \
        --output_suffix "_seed${ENC}" \
        --output_dir "$OUT" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_group t1_multiseed \
        --wandb_tags "encoder_seed${ENC}"

    echo ">>> SNR diagnostic, encoder seed=${ENC}"
    python downstream/snr_ablation.py \
        --embeddings "$EMB" \
        --features   "$SPEC" \
        --seed "$ENC" \
        --output "$OUT/snr_ablation_seed${ENC}.json"
done

# ── LineTransfer : Hβ -> Hα ─────────────────────────────────────────────────
echo ""
echo ">>> LineTransfer  $(date)"
python downstream/t2_cross_line_probing.py \
    --hbeta_features "$HBETA" \
    --halpha_features "$SPEC" \
    --output "$OUT/probe_t2_hbeta_to_halpha.json" \
    --output_fig "$REPO_ROOT/results_reproduced/figures/t2_hbeta_to_halpha.png"

# ── EWForecast : trained in-domain baselines ────────────────────────────────
echo ""
echo ">>> EWForecast PCA + Ridge  $(date)"
python downstream/t3_pca_ridge_temporal_baseline.py \
    --embeddings "$EMB42" \
    --features   "$SPEC" \
    --output_dir "$OUT" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group t3_pca_ridge

for CTX in 5 10; do
    echo ">>> EWForecast z + Ridge, context=${CTX}"
    python downstream/t3_z_ridge_temporal_baseline.py \
        --context "$CTX" \
        --embeddings "$EMB42" \
        --features   "$SPEC" \
        --output_dir "$OUT"
done

export BESS_EMB_PATH="$EMB42" BESS_FEATURES_PATH="$SPEC" BESS_DOWNSTREAM_DIR="$OUT"
echo ">>> EWForecast time-aware ridge"
python downstream/t3_time_aware_ridge.py
echo ">>> EWForecast validation-based selection of the PCA+Ridge configuration"
python downstream/t3_pca_ridge_select_config.py
echo ">>> EW noise floor and PCA scree"
python downstream/compute_sigma_ew_and_scree.py

echo ""
echo "=============================================================="
echo "PHASE 2a done  —  $(date)"
echo "=============================================================="
