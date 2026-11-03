# stage1_encoder/ — BeMAE, the reference Hα encoder

BeMAE is a small masked auto-encoder trained on a 100 Å window centred on
Hα (6562.8 Å). It is the *reference* learned representation of
BESS-Bench, not a methodological contribution in itself.

## Architecture

| Item | Value |
|---|---|
| Input | 128 bins over Hα ± 50 Å, flux normalised to the pseudo-continuum at the window edges |
| Patches | 8 px, stride 4 → 31 patches |
| Encoder | Transformer, 4 layers, d_model = 128, 4 heads, FFN 512 — ≈795 k parameters |
| Decoder (training only) | 2 layers, d = 64 — the full auto-encoder has ≈912 k parameters |
| Objective | masked auto-encoding, mask ratio 0.60 in 3 contiguous blocks, MSE on masked patches |
| Output | 128-dimensional embedding `z_halpha` |

## Data split used for pre-training

Pre-training uses **no label**. The encoder is trained on the Hα spectra
of all stars except **eight held-out stars** (GAM CAS, PLEIONE, ZET TAU,
V442 AND, PI AQR, DEL SCO, 28 CYG, OME CMA; fixed in
`config.py::TrainConfig.test_stars`), which serve for early stopping and
for the held-out diagnostics: 21 781 training spectra of 1 455 stars,
5 156 validation spectra of 8 stars. The 70/15/15 star-level partition
shipped with the dataset (`splits.csv`) is **not** used here; see
`EVALUATION_PROTOCOL.md` §0 for what this implies.

## Files

- `config.py` — configuration (architecture and training)
- `dataset.py` — data pipeline: Hα crop ± 50 Å → 128 interpolated bins
- `model.py` — `SpectralEncoderHalpha` and the MAE decoder
- `train.py` — training loop with early stopping
- `evaluate.py` — latent-space diagnostics (see below)
- `precompute_embeddings.py` — frozen embeddings for the downstream tasks
- `export_hf.py` — encoder export in the layout of the HuggingFace model repository
- `hf_export/bemae-halpha-v1/` — content of the published model repository (three seeds)

## Usage

```bash
cd stage1_encoder

# one seed
python train.py --seed 42 --run_name Halpha_all_seed42
python evaluate.py --checkpoint runs/Halpha_all_seed42/best.pt
python precompute_embeddings.py --checkpoint runs/Halpha_all_seed42/best.pt \
    --output_dir ../data/embeddings_halpha_seed42

# quick local test on CPU
python train.py --epochs 5 --device cpu --batch_size 16 --run_name test_local

# export for the HuggingFace model repository
python export_hf.py --checkpoint runs/Halpha_all_seed42/best.pt \
    --output_dir hf_export/bemae-halpha-v1
```

The three seeds of the paper (42, 123, 456), their diagnostics and their
embeddings are produced by one job: `sbatch scripts/cluster/00_pretrain_encoder.slurm`
(a plain Bash script; the `#SBATCH` lines are comments outside SLURM).

## Diagnostics (`evaluate.py`)

| Key | Meaning |
|---|---|
| `knn_intra_star` | within-star emission/absorption k-NN accuracy, over the stars observed in both states (22 stars) |
| `recall_at_10_test` | star retrieval on the eight held-out stars: fraction of same-star spectra among the 10 nearest neighbours |
| `recall_at_10` | the same over all stars of the slice |
| `r2_cv_ew_*` | per-star cross-validated R² of EW from the embedding |
| `cos_centroids` | cosine similarity between emission and absorption centroids |

Values for the three released checkpoints are in
`reference_results/stage1_encoder/` (`multiseed_summary.json`: mean ±
sample std over the seeds).

## Reproducibility

- **Dataset**: `matthieulel/bess-bench` (Hα slice, built on the fly by `dataset.py`).
- **Hyper-parameters**: `config.py` (up to 80 epochs, early stopping with
  patience 15, mask ratio 0.60, AdamW, lr 1e-4, batch 256).
- **Seeds**: 42 (reference checkpoint), 123, 456.
- **Hardware**: one NVIDIA RTX 2080 Ti per run, 41 to 51 minutes per seed.
- **Released checkpoints**: `matthieulel/bemae-halpha-v1` on
  HuggingFace (seed 42 at the root, `seed123/`, `seed456/`). The seed-42
  file is the one evaluated during peer review. GPU training is not
  bit-reproducible: retraining a seed gives a different checkpoint with
  equivalent scores, so the published weights are the reference for the
  numbers of the paper.
