#!/bin/bash
#SBATCH --job-name=bess-00-encoder
#SBATCH --partition=shortrun,longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/00_encoder_%j.out
#SBATCH --error=logs/00_encoder_%j.err
# ════════════════════════════════════════════════════════════════════════════
# PHASE 0 — BeMAE-Hα encoder: pre-training, diagnostics, embeddings (3 seeds)
# ════════════════════════════════════════════════════════════════════════════
#
# For each seed ∈ {42, 123, 456} (training is skipped when 00a_train_seed.slurm
# has already produced the checkpoint):
#   - stage1_encoder/runs/Halpha_all_seed<S>/best.pt
#   - reference_results/stage1_encoder/Halpha_all_seed<S>/eval_results.json
#   - reference_results/embeddings[/seed<S>]/star_data.pt   (seed 42 at the root)
#
# And once:
#   - reference_results/stage1_encoder/multiseed_summary.json
#   - stage1_encoder/hf_export/bemae-halpha-v1/   (HuggingFace layout, 3 seeds)
#
# Duration : ~2h15 on one RTX 2080 Ti (41 to 51 min per seed) + ~20 min
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/_common.sh"

echo "=============================================================="
echo "PHASE 0 — BeMAE-Hα multi-seed pre-training"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

cd "$REPO_ROOT/stage1_encoder"
mkdir -p runs hf_export

# ── 1. Train + diagnostics + embeddings for each seed ───────────────────────
for SEED in "${SEEDS[@]}"; do
    RUN_NAME="Halpha_all_seed${SEED}"
    RUN_DIR="runs/${RUN_NAME}"
    EVAL_DIR="$REF/stage1_encoder/${RUN_NAME}"
    EMB_DIR="$(emb_dir "$SEED")"

    echo ""
    echo "── seed=${SEED}  |  run=${RUN_NAME} ──────────────────────────"

    if [[ -f "${RUN_DIR}/best.pt" ]]; then
        echo ">>> SKIP train (${RUN_DIR}/best.pt exists)"
    else
        echo ">>> TRAIN seed=${SEED}  $(date)"
        python train.py \
            --run_name "$RUN_NAME" \
            --seed "$SEED" \
            --output_dir runs \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_group bemae_pretrain_v1 \
            --wandb_tags "multiseed"
    fi

    echo ">>> EVAL seed=${SEED}  $(date)"
    python evaluate.py --checkpoint "${RUN_DIR}/best.pt" --output_dir "$EVAL_DIR"
    cp "${RUN_DIR}/config.json" "$EVAL_DIR/config.json"

    echo ">>> EMBEDDINGS seed=${SEED}  $(date)"
    python precompute_embeddings.py \
        --checkpoint "${RUN_DIR}/best.pt" \
        --output_dir "${EMB_DIR}"
done

# ── 2. Multi-seed summary of the diagnostics ────────────────────────────────
echo ""
echo ">>> Aggregating multi-seed diagnostics"
python - "$REF/stage1_encoder" <<'PYEOF'
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = [42, 123, 456]

def pick(d, *keys):
    """Follow a path inside nested dicts; None if a key is missing or the value is not numeric."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None

rows = []
for s in seeds:
    evalp = root / f"Halpha_all_seed{s}" / "eval_results.json"
    if not evalp.exists():
        print(f"  ! missing {evalp}")
        continue
    d = json.loads(evalp.read_text())
    rows.append({
        "seed": s,
        "r2_cv_ew_test":     pick(d, "r2_cv_ew_test", "r2_mean"),
        "r2_cv_ew_all":      pick(d, "r2_cv_ew_all", "r2_mean"),
        "knn_intra_star":    pick(d, "knn_intra_star", "mean_accuracy"),
        "cos_centroids":     pick(d, "cos_centroids", "cos_mean"),
        "recall_at_10_test": pick(d, "recall_at_10_test"),
        "recall_at_10":      pick(d, "recall_at_10"),
    })

def agg(key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n": len(vals)}

metrics = ["r2_cv_ew_test", "r2_cv_ew_all", "knn_intra_star",
           "cos_centroids", "recall_at_10_test", "recall_at_10"]
summary = {"seeds": seeds, "per_seed": rows, "aggregate": {k: agg(k) for k in metrics}}
out = root / "multiseed_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"  -> {out}")
for k, v in summary["aggregate"].items():
    if v.get("mean") is not None:
        print(f"     {k:<24s} {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
PYEOF

# ── 3. Export the encoders in the layout of the HuggingFace model repository ─
echo ""
echo ">>> HuggingFace export  $(date)"
python export_hf.py --checkpoint runs/Halpha_all_seed42/best.pt \
    --output_dir hf_export/bemae-halpha-v1
for SEED in 123 456; do
    python export_hf.py --checkpoint "runs/Halpha_all_seed${SEED}/best.pt" \
        --output_dir "hf_export/bemae-halpha-v1/seed${SEED}" --weights_only
done

echo ""
echo "=============================================================="
echo "PHASE 0 done  —  $(date)"
echo "=============================================================="
