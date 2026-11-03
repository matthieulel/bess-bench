#!/bin/bash
#SBATCH --job-name=bess-03-aggregate
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/03_aggregate_%j.out
#SBATCH --error=logs/03_aggregate_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Aggregation, confidence intervals, tables and figures (CPU)
# ════════════════════════════════════════════════════════════════════════════
#
# Produces :
#   - reference_results/stats/t1_multiseed_summary.json   (SpecProbe, 3 seeds, Eq. 1)
#   - reference_results/stats/ablations_summary.json      (ablation grid)
#   - reference_results/stats/t3_bootstrap_cis.json       (EWForecast CI95)
#   - paper/tables/*.tex, paper/tables/summary.json
#   - paper/figures/*.pdf
#
# Requires Phases 0, 1, 2a, 2b (and 0b + 2c for the ablation table).
# Duration : ~10 min.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 3 — Aggregation, tables and figures"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

cd "$REPO_ROOT"

echo ""
echo ">>> SpecProbe multi-seed summary and headline ratio"
python stats/aggregate_t1_multiseed.py --seeds "${SEEDS[@]}" \
    --wandb_project "$WANDB_PROJECT" --wandb_group t1_multiseed_agg

echo ""
echo ">>> Ablation grid"
python stats/aggregate_ablations.py

echo ""
echo ">>> SpecProbe Gaussian CI95 (per-fold, seed 42)"
python stats/bootstrap_ci.py

echo ""
echo ">>> EWForecast bootstrap CI95"
python stats/bootstrap_ci_t3.py \
    --wandb_project "$WANDB_PROJECT" --wandb_group t3_bootstrap

echo ""
echo ">>> EWForecast on well-sampled / sparsely sampled stars"
python downstream/t3_well_sampled_stratification.py

echo ""
echo ">>> Paper tables (LaTeX)"
python stats/aggregate_results.py

echo ""
echo ">>> Paper figures"
python stats/figure_t3_horizon.py
for fig in fig1_teaser_gallery fig3_embedding_pca fig4_stage1_probes \
           fig5_t1_per_feature fig6_t3_forecast fig_pca_scree; do
    python "paper/figure_scripts/${fig}.py"
done
# Fig. 2 (dataset quality profile) and the coverage audit need a metadata
# cache built from the full dataset:
#     python stats/extract_metadata_slim.py
#     python stats/coverage_audit.py --output reference_results/stats/coverage_audit.json
#     python paper/figure_scripts/fig2_dataset_quality.py

echo ""
echo "=============================================================="
echo "PHASE 3 done  —  $(date)"
echo "=============================================================="
