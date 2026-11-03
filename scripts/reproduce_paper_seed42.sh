#!/usr/bin/env bash
# ==============================================================================
# reproduce_paper_seed42.sh
# ──────────────────────────────────────────────────────────────────────────────
# Reproduce the paper results that depend on the reference encoder (seed 42)
# starting from the PUBLIC artefacts on the HuggingFace Hub:
#
#   model   : https://huggingface.co/matthieulel/bemae-halpha-v1
#   dataset : https://huggingface.co/datasets/matthieulel/bess-bench
#
# All output is written under  results_reproduced/  (this repository, never
# anywhere else). At the end, scripts/compare_with_reference.py compares the
# outputs with the deposited reference_results/.
#
# What this reproduces (seed 42):
#     SpecProbe            (probe_features_results_seed42.json)
#     LineTransfer         (probe_t2_hbeta_to_halpha.json)
#     EWForecast           (t3_pca_ridge_temporal_baseline.json, z-ridge,
#                           t3_time_aware_ridge.json, t3_pca_ridge_select_config.json)
#     EWForecast TS-FMs    (t3_ts_fm_baselines.json: Chronos-Bolt, TimesFM-2.0)
#     SNR diagnostic       (snr_ablation_seed42.json)
#     noise floor / scree  (sigma_ew_and_scree.json)
#
# Options (environment variables):
#     DEVICE=cpu|cuda   device for the encoder and the TS-FMs (default: cuda if available)
#     SKIP_TSFM=1       skip every time-series foundation model
#     ALL_SEEDS=1       also run seeds 123 and 456 (SpecProbe, SNR diagnostic)
#                       and recompute the three-seed headline ratio (Eq. 1)
#     EXTRA_TSFM=1      also run the additional forecasters when their packages
#                       are installed: trained head on frozen Chronos-Bolt,
#                       Time-MoE, Moirai, Moirai-MoE (see requirements.txt)
#     MODEL_DIR=<dir>   use a local copy of the model repository (no download)
#
# What this DOES NOT re-run: encoder pre-training and the 24 ablation
# checkpoints. For these, see scripts/cluster/run_all.sh. Their reference
# numbers are deposited in reference_results/.
#
# Determinism:
#   - WANDB_MODE=disabled -> no telemetry
#   - PYTHONHASHSEED, OMP/MKL pinned
#   - the probes use star-grouped folds (GroupKFold), which are deterministic
#   - device = cuda matches the reference run (RTX 2080 Ti); on CPU the
#     differences stay below the tolerances of the comparison report
#
# Usage (from the repository root):
#     bash scripts/reproduce_paper_seed42.sh                 # GPU (default)
#     DEVICE=cpu bash scripts/reproduce_paper_seed42.sh      # CPU fallback
#     ALL_SEEDS=1 bash scripts/reproduce_paper_seed42.sh     # three seeds
#
# Total runtime: ~20-30 min on one RTX 2080 Ti (seed 42, incl. TS-FMs).
# ==============================================================================

set -euo pipefail

# ---- trap: print which line failed -----------------------------------------
_on_error() {
    local exit_code=$?
    local line_no=$1
    echo ""
    echo "========================================================"
    echo "ERROR: script failed at line ${line_no} (exit code ${exit_code})"
    echo "Check the log above for the Python traceback."
    echo "========================================================"
}
trap '_on_error $LINENO' ERR

# ---- repo root (assumes this script lives in <repo>/scripts/) ---------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# ---- determinism / no telemetry -----------------------------------------------
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=false

# DEVICE: default to cuda (matches reference run); fallback to cpu if no GPU
if [[ -z "${DEVICE:-}" ]]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE=cuda
    else
        DEVICE=cpu
        echo "[INFO] CUDA not available; falling back to DEVICE=cpu"
    fi
fi
SKIP_TSFM="${SKIP_TSFM:-0}"      # 1 -> skip every time-series foundation model
ALL_SEEDS="${ALL_SEEDS:-0}"      # 1 -> also run encoder seeds 123 and 456
EXTRA_TSFM="${EXTRA_TSFM:-0}"    # 1 -> also run Chronos head, Time-MoE, Moirai, Moirai-MoE
MODEL_DIR="${MODEL_DIR:-}"       # local copy of the model repository (optional)

# ---- output layout -----------------------------------------------------------
OUT="$REPO/results_reproduced"
mkdir -p \
    "$OUT/assets/hf_model" \
    "$OUT/embeddings" \
    "$OUT/features" \
    "$OUT/downstream" \
    "$OUT/figures" \
    "$OUT/logs"

LOG="$OUT/logs/reproduce.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================================="
echo "BESS-Bench reproduction (seed 42)"
echo "Date     : $(date)"
echo "Repo     : $REPO"
echo "Output   : $OUT"
echo "Device   : $DEVICE"
echo "Threads  : OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS"
echo "Skip TSFM: $SKIP_TSFM   All seeds: $ALL_SEEDS   Extra TS-FMs: $EXTRA_TSFM"
echo "======================================================================="

# ---- ensure huggingface_hub is available -------------------------------------
python -c "import huggingface_hub" 2>/dev/null || {
    echo ">>> Installing huggingface_hub..."
    pip install --quiet huggingface_hub
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Download the public encoder checkpoint(s)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 1 / 7 -- Encoder checkpoint(s) from the HuggingFace Hub"
echo "======================================================================"
CKPT="$OUT/assets/hf_model/pytorch_model.bin"
if [[ -n "$MODEL_DIR" ]]; then
    echo "    using local model directory: $MODEL_DIR"
    cp "$MODEL_DIR/pytorch_model.bin" "$MODEL_DIR/config.json" "$OUT/assets/hf_model/"
    if [[ "$ALL_SEEDS" == "1" ]]; then
        for s in 123 456; do
            mkdir -p "$OUT/assets/hf_model/seed$s"
            cp "$MODEL_DIR/seed$s/pytorch_model.bin" "$MODEL_DIR/seed$s/config.json" \
               "$OUT/assets/hf_model/seed$s/"
        done
    fi
elif [[ -s "$CKPT" && -s "$OUT/assets/hf_model/config.json" \
        && ( "$ALL_SEEDS" != "1" || -s "$OUT/assets/hf_model/seed456/pytorch_model.bin" ) ]]; then
    echo "    already present -> $CKPT  ($(du -h "$CKPT" | cut -f1))"
else
    ALL_SEEDS="$ALL_SEEDS" python - <<PY
import os
from huggingface_hub import snapshot_download
patterns = ["pytorch_model.bin", "config.json"]
if os.environ.get("ALL_SEEDS") == "1":
    patterns += ["seed123/*", "seed456/*"]
p = snapshot_download(
    repo_id="matthieulel/bemae-halpha-v1",
    local_dir="$OUT/assets/hf_model",
    allow_patterns=patterns,
)
print("downloaded to:", p)
PY
fi
ls -la "$OUT/assets/hf_model/"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Extract embeddings z ∈ ℝ¹²⁸ from the encoder
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 2 / 7 -- Extracting embeddings (device=$DEVICE)"
echo "======================================================================"
EMB_FILE="$OUT/embeddings/star_data.pt"
if [[ -s "$EMB_FILE" ]]; then
    echo "    already present -> $EMB_FILE"
else
    python stage1_encoder/precompute_embeddings.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUT/embeddings" \
        --device "$DEVICE" \
        --batch_size 128 \
        --num_workers 2
fi
if [[ "$ALL_SEEDS" == "1" ]]; then
    for s in 123 456; do
        if [[ -s "$OUT/embeddings/seed$s/star_data.pt" ]]; then
            echo "    already present -> $OUT/embeddings/seed$s/star_data.pt"
        else
            python stage1_encoder/precompute_embeddings.py \
                --checkpoint "$OUT/assets/hf_model/seed$s/pytorch_model.bin" \
                --output_dir "$OUT/embeddings/seed$s" \
                --device "$DEVICE" \
                --batch_size 128 \
                --num_workers 2
        fi
    done
fi
ls -la "$OUT/embeddings/"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Compute classical spectral features (FWHM, EW, V/R, …)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 3 / 7 -- Classical spectral features (Halpha + Hbeta)"
echo "======================================================================"
SPEC="$OUT/features/spectral_features.pt"
HBETA="$OUT/features/spectral_features_hbeta.pt"
if [[ -s "$SPEC" ]]; then
    echo "    already present -> $SPEC"
else
    python downstream/compute_spectral_features.py --output "$SPEC"
fi
if [[ -s "$HBETA" ]]; then
    echo "    already present -> $HBETA"
else
    python downstream/compute_hbeta_features.py --output "$HBETA"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- SpecProbe (ridge probes -> 6 features, star-grouped CV), seed 42
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 4 / 7 -- SpecProbe (seed 42)"
echo "======================================================================"
# --seed is the seed of the probe script (PCA baselines); the star-grouped
# folds do not depend on it. reference_results/ was produced with 0.
python downstream/probe_features.py \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --seed 0 \
    --output_suffix "_seed42" \
    --output_dir "$OUT/downstream" \
    --figures_dir "$OUT/figures" \
    --wandb_mode disabled
if [[ "$ALL_SEEDS" == "1" ]]; then
    for s in 123 456; do
        python downstream/probe_features.py \
            --embeddings "$OUT/embeddings/seed$s/star_data.pt" \
            --features   "$SPEC" \
            --seed 0 \
            --output_suffix "_seed$s" \
            --output_dir "$OUT/downstream" \
            --figures_dir "$OUT/figures" \
            --wandb_mode disabled
    done
    mkdir -p "$OUT/stats"
    python stats/aggregate_t1_multiseed.py \
        --input-dir "$OUT/downstream" \
        --output    "$OUT/stats/t1_multiseed_summary.json" \
        --wandb_mode disabled
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- LineTransfer (Hbeta window -> Halpha features, star-grouped CV)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 5 / 7 -- LineTransfer"
echo "======================================================================"
python downstream/t2_cross_line_probing.py \
    --hbeta_features "$HBETA" \
    --halpha_features "$SPEC" \
    --output "$OUT/downstream/probe_t2_hbeta_to_halpha.json" \
    --output_fig "$OUT/figures/t2_hbeta_to_halpha.png"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- EWForecast: PCA+Ridge, z-Ridge, time-aware ridge, TS foundation models
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 6 / 7 -- EWForecast"
echo "======================================================================"
python downstream/t3_pca_ridge_temporal_baseline.py \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --output_dir "$OUT/downstream" \
    --seed 42 \
    --wandb_mode disabled

echo ""
echo "    --- z-Ridge on encoder embeddings (context = 5 and 10) ---"
for ctx in 5 10; do
    python downstream/t3_z_ridge_temporal_baseline.py \
        --context    "$ctx" \
        --embeddings "$EMB_FILE" \
        --features   "$SPEC" \
        --output_dir "$OUT/downstream"
done

# The scripts below read their inputs / output directory from the environment.
export BESS_EMB_PATH="$EMB_FILE"
export BESS_FEATURES_PATH="$SPEC"
export BESS_DOWNSTREAM_DIR="$OUT/downstream"

echo ""
echo "    --- time-aware and context-statistic ridge baselines (CPU) ---"
python downstream/t3_time_aware_ridge.py
echo ""
echo "    --- validation-based selection of the PCA+Ridge configuration ---"
python downstream/t3_pca_ridge_select_config.py
echo ""
echo "    --- EW measurement noise floor and PCA scree ---"
python downstream/compute_sigma_ew_and_scree.py

if [[ "$SKIP_TSFM" == "1" ]]; then
    echo ""
    echo "    [skipped] time-series foundation models  (SKIP_TSFM=1)"
else
    python downstream/t3_ts_fm_baselines.py \
        --embeddings "$EMB_FILE" \
        --features   "$SPEC" \
        --output_dir "$OUT/downstream" \
        --device "$DEVICE" \
        --wandb_mode disabled
    if [[ "$EXTRA_TSFM" == "1" ]]; then
        run_if_importable() {   # <python module> <script>
            if python -c "import $1" 2>/dev/null; then
                echo ""
                echo "    --- $2 ---"
                python "downstream/$2"
            else
                echo "    [skipped] $2  (python package '$1' not installed)"
            fi
        }
        run_if_importable chronos      t3_chronos_zeroshot_samepairs.py
        run_if_importable chronos      t3_chronos_trained_head.py
        run_if_importable transformers t3_timemoe_zeroshot.py
        run_if_importable uni2ts       t3_moirai_zeroshot.py
        run_if_importable uni2ts       t3_moiraimoe_zeroshot.py
    fi
fi
echo ""
echo "    --- EWForecast on well-sampled / sparsely sampled stars ---"
python downstream/t3_well_sampled_stratification.py
unset BESS_EMB_PATH BESS_FEATURES_PATH BESS_DOWNSTREAM_DIR

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- SpecProbe per-SNR-bin diagnostic (Appendix)
# ─────────────────────────────────────────────────────────────────────────────
# Star-grouped folds are deterministic: the seed is the ENCODER seed, i.e.
# the checkpoint the embeddings come from.
echo ""
echo "======================================================================"
echo "STEP 7 / 7 -- SpecProbe per-SNR-bin diagnostic"
echo "======================================================================"
python downstream/snr_ablation.py \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --seed 42 \
    --output "$OUT/downstream/snr_ablation_seed42.json"
if [[ "$ALL_SEEDS" == "1" ]]; then
    for s in 123 456; do
        python downstream/snr_ablation.py \
            --embeddings "$OUT/embeddings/seed$s/star_data.pt" \
            --features   "$SPEC" \
            --seed "$s" \
            --output "$OUT/downstream/snr_ablation_seed${s}.json"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "REPRODUCTION COMPLETE"
echo "======================================================================="
echo "Output written to:  $OUT"
echo ""

# ---- regenerate paper figures -----------------------------------------
# All figures are written under  $OUT/figures/  (results_reproduced/figures/).
# The committed canonical copies under  paper/figures/  are NEVER overwritten.
echo "======================================================================="
echo "Regenerating paper figures into $OUT/figures/"
echo "======================================================================="
mkdir -p "$OUT/figures"

# Fig. T3-horizon: depends on freshly produced t3_*.json from $OUT/downstream/.
echo "----- Fig. T3-horizon (from reproduced t3_* JSONs) -----"
python stats/figure_t3_horizon.py \
    --input-dir "$OUT/downstream" \
    --output    "$OUT/figures/fig_t3_horizon.pdf"

# Fig. 1 (teaser gallery) and Fig. 3 (embedding PCA): consume freshly produced
# embeddings + features from $OUT (true seed-42 reproduction).
export BESS_FIG_DIR="$OUT/figures"
export BESS_FEATURES_PATH="$SPEC"
export BESS_EMB_PATH="$EMB_FILE"

echo "----- Fig. 1  (teaser gallery, 5 morphological classes) -----"
python paper/figure_scripts/fig1_teaser_gallery.py

echo "----- Fig. 3  (embedding PCA coloured by morphology) -----"
python paper/figure_scripts/fig3_embedding_pca.py

# Figs. 4, 5, 6: re-render committed multi-seed numbers (reference_results/stats/).
# Multi-seed inputs (t1_multiseed_summary.json, t3_bootstrap_cis.json) are
# produced by Path B only; path A reuses the deposited JSONs to verify that
# the plotting pipeline itself is deterministic.
echo "----- Fig. 4  (Stage-1 probes) -----"
python paper/figure_scripts/fig4_stage1_probes.py

echo "----- Fig. 5  (T1 SpecProbe per-feature R^2) -----"
python paper/figure_scripts/fig5_t1_per_feature.py

echo "----- Fig. 6  (T3 forecast bootstrap) -----"
python paper/figure_scripts/fig6_t3_forecast.py

echo "----- PCA scree (appendix; from the reproduced features) -----"
python paper/figure_scripts/fig_pca_scree.py

unset BESS_FIG_DIR BESS_FEATURES_PATH BESS_EMB_PATH

# ---- regenerate paper tables ------------------------------------------
# Same convention: written under  $OUT/tables/ , committed copies under
# paper/tables/  are NEVER overwritten.
echo "======================================================================="
echo "Regenerating paper tables into $OUT/tables/"
echo "======================================================================="
mkdir -p "$OUT/tables"

export BESS_TABLES_DIR="$OUT/tables"
export BESS_DOWNSTREAM_DIR="$OUT/downstream"

# Aggregate Stage-1 multi-seed (reads committed reference_results/stage1_encoder/),
# T1 probes (reads $OUT/downstream/probe_features_results_seed42.json), and
# T3 baselines (reads $OUT/downstream/t3_*.json).
python stats/aggregate_results.py

# T1 95% CI from CV folds.
python stats/bootstrap_ci.py

unset BESS_TABLES_DIR BESS_DOWNSTREAM_DIR

# Note on Fig. 2 (dataset quality profile):
#   Requires  paper/tables/metadata_slim.parquet  built from the full HF
#   snapshot via  stats/extract_metadata_slim.py  (~2 GB download, ~1 min).
#   Skipped from the quick reproduction; the canonical PDF/PNG ships under
#   paper/figures/ for inspection. To regenerate locally:
#       python stats/extract_metadata_slim.py
#       python paper/figure_scripts/fig2_dataset_quality.py
echo ""

find "$OUT" -type f \( -name "*.json" -o -name "*.pt" -o -name "*.pdf" -o -name "*.png" \) 2>/dev/null \
    | sort | sed 's|'"$REPO"'/|    |'
echo ""

# ---- automatic comparison against reference_results/ ----------------------
SKIP_COMPARE="${SKIP_COMPARE:-0}"
if [[ "$SKIP_COMPARE" != "1" && -d "$REPO/reference_results" ]]; then
    echo "======================================================================="
    echo "AUTOMATIC COMPARISON AGAINST reference_results/"
    echo "======================================================================="
    python scripts/compare_with_reference.py \
        --tol-strict 1e-3 \
        --tol-gpu    1e-2 \
        || echo "[INFO] compare_with_reference.py reported deviations; see report above and REPRODUCE.md."
    echo "======================================================================="
else
    echo "To compare against reference numbers, run:"
    echo "    python scripts/compare_with_reference.py"
    echo "======================================================================="
fi
