#!/bin/bash
#SBATCH --job-name=bess-02c-ablations-probe
#SBATCH --partition=shortrun,longrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/02c_ablations_probe_%A_%a.out
#SBATCH --error=logs/02c_ablations_probe_%A_%a.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 2c — SpecProbe on the ablation encoders (CPU)
# ════════════════════════════════════════════════════════════════════════════
#
# For each ablation variant [0..7], runs probe_features.py on the 3 encoder
# seeds {42, 123, 456}, restricted to the 3 headline features
# {fwhm, central_depth, delta_v}. The last array task to finish can be
# followed by  python stats/aggregate_ablations.py  (done in Phase 3).
#
# Output : reference_results/downstream/probe_features_results_ablation_<tag>_seed<S>.json
#
# Requires Phase 0b (ablation embeddings) and Phase 1 (features).
# Without SLURM :  SLURM_ARRAY_TASK_ID=<0..7> bash 02c_ablations_probe.slurm
# Duration : ~10 min/job on 8 CPU cores.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

TAG="${ABLATION_VARIANTS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================================="
echo "PHASE 2c — SpecProbe on ablation encoders : tag=${TAG}"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

cd "$REPO_ROOT"

for ENC in "${SEEDS[@]}"; do
    EMB_FILE="$REPO_ROOT/data/embeddings_ablation_${TAG}_seed${ENC}/star_data.pt"
    if [[ ! -f "$EMB_FILE" ]]; then
        echo "WARNING: missing embeddings ${EMB_FILE}, skipping."
        continue
    fi
    echo ">>> PROBE ${TAG} seed=${ENC}  $(date)"
    python downstream/probe_features.py \
        --embeddings "$EMB_FILE" \
        --features   "$SPEC" \
        --feature_list fwhm central_depth delta_v \
        --seed "$PROBE_SEED" \
        --output_suffix "_ablation_${TAG}_seed${ENC}" \
        --output_dir "$REF/downstream" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_group t1_ablations \
        --wandb_tags "ablation" "${TAG}" "encoder_seed${ENC}"
done

echo ""
echo "PHASE 2c done : ${TAG}  $(date)"
