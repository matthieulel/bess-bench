#!/bin/bash
#SBATCH --job-name=bess-01-features
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/01_features_%j.out
#SBATCH --error=logs/01_features_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Spectral feature extraction (CPU)
# ════════════════════════════════════════════════════════════════════════════
#
# Extracts the 6 Hα spectral features (EW, FWHM, V/R, Δv, depth, peak
# intensity) and the profile-type labels from the full HuggingFace dataset,
# plus the Hβ window needed by LineTransfer.
#
# Outputs :
#   - reference_results/features/spectral_features.pt         (Hα)
#   - reference_results/features/spectral_features_hbeta.pt   (Hβ)
#
# Independent of Phase 0 — can run in parallel with it.
# Duration : ~30 min to 1 h (reads the full HF dataset twice).
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 1 — Spectral feature extraction"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

cd "$REPO_ROOT"

echo ">>> Extracting Hα features  $(date)"
python downstream/compute_spectral_features.py --output "$SPEC"

echo ">>> Extracting Hβ features  $(date)"
python downstream/compute_hbeta_features.py --output "$HBETA"

echo ""
echo "PHASE 1 done  —  $(date)"
