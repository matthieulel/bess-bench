#!/bin/bash
# =============================================================================
# export_to_hf.sh — Export BeSS → HuggingFace Hub (via SLURM)
# =============================================================================
# Comment/uncomment the steps as needed, then: sbatch export_to_hf.sh
# Follow with: tail -f ../../logs/export_hf_<jobid>.out
# =============================================================================

#SBATCH --job-name=bess-hf-export
#SBATCH --output=../../logs/export_hf_%j.out
#SBATCH --error=../../logs/export_hf_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G



# ── Configuration ────────────────────────────────────────────────────────────
HF_REPO_ID="${HF_REPO_ID:-<user>/bess-bench-26}"
N_WORKERS=${SLURM_CPUS_PER_TASK:-8}
# EXTRACT_LIMIT=100   # Uncomment for a quick test

# ── Environment ──────────────────────────────────────────────────────────────
set -euo pipefail
PROJECT_DIR="."
EXPORT_DIR="${PROJECT_DIR}/data_pipeline/03_export"
cd "${EXPORT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Load .env
[[ -f "${PROJECT_DIR}/.env" ]] && { set -a; source "${PROJECT_DIR}/.env"; set +a; }

# Conda activation (adjust to your local installation)
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
        # shellcheck disable=SC1090
        source "$candidate"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found"; exit 1
fi
conda activate "${CONDA_ENV:-base}"

echo "=== EXPORT BESS | $(date -Iseconds) | Job ${SLURM_JOB_ID:-local} ==="

# ── Step 1: FITS → Parquet (~30-60 min) ─────────────────────────────────────
#python 01_extract_spectra.py --workers ${N_WORKERS} ${EXTRACT_LIMIT:+--limit $EXTRACT_LIMIT}

# ── Step 2: assemble HuggingFace DatasetDict (~15-30 min) ───────────────────
#python 02_build_dataset.py

# ── Step 3: push to HuggingFace Hub (~30-60 min) ────────────────────────────
python 03_push_to_hub.py --repo-id "${HF_REPO_ID}" --private

echo "=== DONE | $(date -Iseconds) ==="
