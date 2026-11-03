#!/bin/bash
# Common settings of the cluster pipeline; sourced by every phase script.
#
#   BESS_CONDA_ENV   conda environment to activate (default: use the current one)
#   WANDB_MODE       Weights & Biases mode (default: disabled)
#   WANDB_PROJECT    Weights & Biases project (default: bess-bench)
#
# The pipeline writes its outputs where the analysis scripts read them by
# default, i.e. it REGENERATES reference_results/ IN PLACE (work on a branch
# or a copy; `git diff reference_results` then shows how the new run differs
# from the numbers of the paper). Checkpoints go to stage1_encoder/runs/ and
# the ablation embeddings to data/.

# Under sbatch, submit from scripts/cluster/ (run_all.sh does it).
CLUSTER_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$CLUSTER_DIR/../.." && pwd)"
REF="$REPO_ROOT/reference_results"

if [[ ! -d "$REPO_ROOT/stage1_encoder" ]]; then
    echo "ERROR: repository root not found from $CLUSTER_DIR (submit from scripts/cluster/)" >&2
    exit 1
fi

if [[ -n "${BESS_CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$BESS_CONDA_ENV"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-bess-bench}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Seed of the probe scripts (PCA baselines). Star-grouped folds do not depend
# on it. The files of reference_results/ were produced with 0.
PROBE_SEED=0

SEEDS=(42 123 456)
ABLATION_VARIANTS=(
    mask_ratio_030
    mask_ratio_075
    n_layers_2
    n_layers_6
    d_model_64
    d_model_256
    patch_size_4
    patch_size_16
)

# Embeddings of the three encoder seeds: seed 42 at the root, others in seed<S>/.
emb_dir() {
    if [[ "$1" == "42" ]]; then echo "$REF/embeddings"; else echo "$REF/embeddings/seed$1"; fi
}
SPEC="$REF/features/spectral_features.pt"
HBETA="$REF/features/spectral_features_hbeta.pt"

mkdir -p "$CLUSTER_DIR/logs" "$REF/embeddings" "$REF/features" "$REF/downstream" \
         "$REF/stats" "$REF/stage1_encoder"
