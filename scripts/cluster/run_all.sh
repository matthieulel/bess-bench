#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all.sh — End-to-end pipeline launcher for BESS-Bench v1.0 (SLURM)
# ══════════════════════════════════════════════════════════════════════════════
#
# Submits every phase with dependencies so that the cluster runs them in the
# correct order:
#
#   Phase 0a (3 × train) ─► Phase 0 (diagnostics, embeddings) ─┐
#                                                              ├─► Phase 2a (CPU downstream)
#   Phase 1 (features) ────────────────────────────────────────┤
#                                                              ├─► Phase 2b (GPU TS-FMs)
#                                                              │
#   [WITH_ABLATIONS=1]  Phase 0b (24 × train) ─► Phase 2c ─────┴─► Phase 3 (aggregation)
#
# Usage :
#   cd scripts/cluster
#   bash run_all.sh                      # three seeds, all tasks
#   WITH_ABLATIONS=1 bash run_all.sh     # also retrain and probe the 24 ablation encoders
#
# Settings (conda environment, W&B) are in _common.sh; adapt the #SBATCH
# partition names of the *.slurm files to your cluster.
#
# The pipeline REGENERATES reference_results/ and paper/tables/ in place:
# run it on a branch or a copy, then `git diff reference_results` shows how
# your run differs from the numbers of the paper. Retraining an encoder on a
# GPU does not give bit-identical weights, so small differences are expected.
# Logs go to scripts/cluster/logs/.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

WITH_ABLATIONS="${WITH_ABLATIONS:-0}"

echo "=============================================================="
echo "BESS-Bench v1.0 — full pipeline launcher"
echo "Date : $(date)"
echo "=============================================================="

# ── Phase 0a : parallel multi-seed training (one GPU job per seed) ───────────
JOB_T42=$(sbatch  --parsable --export=ALL,SEED=42  --job-name=bess-00a-s42  00a_train_seed.slurm)
JOB_T123=$(sbatch --parsable --export=ALL,SEED=123 --job-name=bess-00a-s123 00a_train_seed.slurm)
JOB_T456=$(sbatch --parsable --export=ALL,SEED=456 --job-name=bess-00a-s456 00a_train_seed.slurm)
echo "submitted  Phase 0a (train seeds 42 / 123 / 456)       → $JOB_T42 $JOB_T123 $JOB_T456"

# ── Phase 0 : diagnostics, embeddings, export (training is skipped) ──────────
JOB0=$(sbatch --parsable --dependency=afterok:${JOB_T42}:${JOB_T123}:${JOB_T456} 00_pretrain_encoder.slurm)
echo "submitted  Phase 0  (diagnostics + embeddings)         → $JOB0"

# ── Phase 1 : feature extraction (CPU, independent) ──────────────────────────
JOB1=$(sbatch --parsable 01_extract_features.slurm)
echo "submitted  Phase 1  (feature extraction)               → $JOB1"

# ── Phase 2 : downstream tasks ───────────────────────────────────────────────
JOB2A=$(sbatch --parsable --dependency=afterok:${JOB0}:${JOB1} 02a_downstream_cpu.slurm)
echo "submitted  Phase 2a (downstream, CPU)                  → $JOB2A"
JOB2B=$(sbatch --parsable --dependency=afterok:${JOB0}:${JOB1} 02b_ts_fm_gpu.slurm)
echo "submitted  Phase 2b (time-series foundation models)    → $JOB2B"

DEPS="${JOB0}:${JOB1}:${JOB2A}:${JOB2B}"
ALL_JOBS="$JOB_T42 $JOB_T123 $JOB_T456 $JOB0 $JOB1 $JOB2A $JOB2B"

# ── Optional : ablation grid ─────────────────────────────────────────────────
if [[ "$WITH_ABLATIONS" == "1" ]]; then
    JOB0B=$(sbatch --parsable 00b_ablations_pretrain.slurm)
    JOB2C=$(sbatch --parsable --dependency=afterok:${JOB0B}:${JOB1} 02c_ablations_probe.slurm)
    echo "submitted  Phase 0b (24 ablation trainings)            → $JOB0B"
    echo "submitted  Phase 2c (ablation probes)                  → $JOB2C"
    DEPS="${DEPS}:${JOB2C}"
    ALL_JOBS="$ALL_JOBS $JOB0B $JOB2C"
fi

# ── Phase 3 : aggregation, tables, figures ───────────────────────────────────
JOB3=$(sbatch --parsable --dependency=afterok:${DEPS} 03_aggregate_paper.slurm)
echo "submitted  Phase 3  (aggregation + paper tables)       → $JOB3"

echo ""
echo "=============================================================="
echo "All jobs submitted. Monitor with :  squeue -u \$USER"
echo ""
echo "Final artefacts :"
echo "   reference_results/                 (regenerated in place)"
echo "   paper/tables/*.tex, summary.json   (paper tables)"
echo "   paper/figures/                     (paper figures)"
echo "   stage1_encoder/hf_export/          (encoders in HuggingFace layout)"
echo ""
echo "To abort :  scancel $ALL_JOBS $JOB3"
echo "=============================================================="
