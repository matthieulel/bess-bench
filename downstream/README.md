# Downstream — BESS-Bench v1.0 evaluation

Evaluation tasks on the Hα slice of BESS-Bench (26 937 spectra covering
Hα ± 50 Å; 26 858 scored spectra of 1 428 stars after the SNR ≥ 10 and
window-coverage gates), for the BeMAE embeddings `z ∈ ℝ¹²⁸` and the
reference baselines.

Official protocol: see `EVALUATION_PROTOCOL.md` at the repository root.

---

## Structure

```
downstream/
├── compute_spectral_features.py      # Prerequisite — Hα features (EW, FWHM, V/R, Δv, depth, type)
├── compute_hbeta_features.py         # Prerequisite — normalised Hβ flux (LineTransfer only)
│
│  ── Benchmarked tasks (v1.0) ─
├── probe_features.py                 # SpecProbe    — ridge probe -> 6 Hα features (star-grouped CV)
├── t2_cross_line_probing.py          # LineTransfer — ridge probe Hβ -> Hα features (star-grouped CV)
├── t3_pca_ridge_temporal_baseline.py # EWForecast   — PCA(K)+Ridge
├── t3_z_ridge_temporal_baseline.py   # EWForecast   — ridge on BeMAE embeddings
├── t3_ts_fm_baselines.py             # EWForecast   — Chronos-Bolt, TimesFM-2.0 (zero-shot)
│
│  ── Additional EWForecast baselines ─
├── t3_moirai_zeroshot.py             # Moirai-1.1-R base / large (zero-shot; needs `uni2ts`)
├── t3_moiraimoe_zeroshot.py          # Moirai-MoE-1.0-R small / base (zero-shot; needs `uni2ts`)
├── t3_timemoe_zeroshot.py            # Time-MoE 50M / 200M (zero-shot)
├── t3_chronos_zeroshot_samepairs.py  # Chronos-Bolt-base restricted to the ctx=5 test pairs
├── t3_chronos_trained_head.py        # trained linear head on frozen Chronos-Bolt forecasts
├── t3_time_aware_ridge.py            # ridge variants that see the elapsed time / context statistics
├── t3_pca_ridge_select_config.py     # validation-based selection of the PCA+Ridge configuration
├── t3_well_sampled_stratification.py # stored predictions split into well-sampled / other stars
│
│  ── Diagnostics ─
├── snr_ablation.py                   # SpecProbe per-SNR-bin diagnostic (appendix)
├── compute_sigma_ew_and_scree.py     # EW measurement noise floor, PCA cumulative variance
└── validate_morphology_audit.py      # consistency check of the morphology labels
```

---

## Prerequisites

1. **Pre-computed embeddings** (one file per encoder seed), produced by
   `stage1_encoder/precompute_embeddings.py` from a BeMAE checkpoint.
   The embeddings used in the paper ship in `reference_results/embeddings/`
   (seed 42 at the root, `seed123/`, `seed456/`).
2. **Dataset**: `matthieulel/bess-bench` (HuggingFace).
3. **Environment**: `pip install -r requirements.txt`.

---

## Execution

The simplest entry point is the reproduction script (seed 42, from the
public artefacts): `bash scripts/reproduce_paper_seed42.sh`. The cluster
pipeline is in `scripts/cluster/`. Manual execution, from the repository
root:

```bash
python downstream/compute_spectral_features.py       # ~20-30 min (reads the HF dataset)
python downstream/compute_hbeta_features.py          # ~5 min
python downstream/probe_features.py                  # SpecProbe, ~2 min
python downstream/t2_cross_line_probing.py           # LineTransfer, ~2 min
python downstream/t3_pca_ridge_temporal_baseline.py  # EWForecast baseline, ~3 min
python downstream/t3_z_ridge_temporal_baseline.py    # EWForecast z+Ridge, ~2 min
python downstream/t3_ts_fm_baselines.py              # EWForecast TS-FMs, ~15 min (GPU)
```

Scripts without command-line options read `BESS_EMB_PATH`,
`BESS_FEATURES_PATH` and `BESS_DOWNSTREAM_DIR` from the environment
(defaults: the files of `reference_results/`).

---

## Benchmarked tasks (v1.0)

### SpecProbe (`probe_features.py`)
RidgeCV probe, 5-fold `GroupKFold` on the star identifier, encoder seeds
42 / 123 / 456: `z -> {EW, FWHM, V/R, Δv, central_depth, peak_intensity}`.
Baselines with the same probe and folds: PCA(10), PCA(32), PCA(50) and
the raw 128-bin flux. Metric: R², mean ± std across encoder seeds.
`--ungrouped` restores the spectrum-level folds of the submitted
manuscript (kept for comparison only: stellar identity leaks across
folds).

### LineTransfer (`t2_cross_line_probing.py`)
RidgeCV probe from the Hβ window (PCA(10), PCA(50) or raw flux) to the Hα
features, on the 2 525 same-night (star, MJD) pairs with both lines
(1 090 stars), 5-fold `GroupKFold` on the star identifier.
Question: which Hα features can be predicted from Hβ alone?

### EWForecast (`t3_*.py`)
One-step prediction of EW(Hα) at the next observation from a context of
L ∈ {5, 10} past observations. Temporal split at MJD 59215.0 (train <
cutoff, test ≥ cutoff). Metric: `ratio_MAE = MAE(model) /
MAE(persistence)` on the same test pairs, with a bootstrap CI95; a model
beats persistence when the upper bound is below 1.
Test population: 7 313 pairs from 621 stars (L = 5), 6 664 pairs from
432 stars (L = 10); zero-shot forecasters are scored on 8 118 pairs from
846 stars.
