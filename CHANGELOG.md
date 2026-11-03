# Changelog

## v1.0 — camera-ready release (2026)

The dataset files are byte-identical to the version that was peer-reviewed.
The code and the reported numbers changed as follows since the submitted
manuscript.

### Evaluation

- **Star-grouped cross-validation for SpecProbe, LineTransfer and the SNR
  diagnostic.** `EVALUATION_PROTOCOL.md` specified 5-fold `GroupKFold` on
  the star identifier, but `downstream/probe_features.py`,
  `downstream/t2_cross_line_probing.py` and `downstream/snr_ablation.py`
  used spectrum-level `KFold(shuffle=True)`, which lets spectra of the same
  star fall on both sides of a fold. The scripts now follow the protocol
  (`--ungrouped` restores the previous behaviour). Effect: lower absolute
  R² values (for instance `fwhm` 0.57 → 0.51 for BeMAE and 0.13 → 0.09 for
  PCA(10)), unchanged ranking; the headline ratio of mean R² over the three
  shape features is 7.77 [7.26, 8.06] (4.24 [4.12, 4.33] with the leaky
  folds, because the linear baselines lose more than the encoder). The
  outputs of the submitted version are kept in
  `reference_results/submission_ungrouped/`.
- **Raw-flux control.** SpecProbe tables and figures now include the ridge
  probe on the raw 128-bin flux next to the PCA baselines.
- **SNR diagnostic.** The PCA(10) baseline of `snr_ablation.py` is computed
  as in SpecProbe (no standardisation before the PCA).
- **Additional EWForecast baselines**, run during the review:
  Moirai-1.1-R, Moirai-MoE-1.0-R and Time-MoE zero-shot
  (`t3_moirai_zeroshot.py`, `t3_moiraimoe_zeroshot.py`,
  `t3_timemoe_zeroshot.py`), a trained linear head on frozen Chronos-Bolt
  forecasts (`t3_chronos_trained_head.py`, with its same-pairs zero-shot
  control `t3_chronos_zeroshot_samepairs.py`), time-aware ridge variants
  (`t3_time_aware_ridge.py`) and the EW measurement noise floor
  (`compute_sigma_ew_and_scree.py`).
- **Standard deviations** across the three encoder seeds (SpecProbe,
  encoder diagnostics, ablations) are sample standard deviations
  (ddof = 1). The causal-Transformer row keeps the value reported during
  the review, 1.17 ± 0.05, which is the population standard deviation
  over its three head seeds (sample value 0.056; range 1.11–1.22).
- **Causal Transformer head.** The code of the Transformer row of the
  EWForecast table is now in `stage2_transformer_head/`, with its caveats
  (own evaluation population; early stopping on the post-cutoff loss).
- **Deterministic PCA.** The PCA baselines of `t2_cross_line_probing.py`,
  `snr_ablation.py` and `compute_sigma_ew_and_scree.py` now use a fixed
  random state (they were unseeded, which moved the LineTransfer PCA(50)
  cells by up to 1e-3 from run to run). PCA(32) and PCA(50) rely on a
  randomized SVD: across library versions their cells still vary at the
  1e-4 level.

### Encoder

- **Three seeds released.** The model repository now holds the checkpoints
  of seeds 42, 123 and 456. Seed 42 is the file evaluated during peer
  review. Seeds 123 and 456 were retrained from this code for the release
  (GPU training is not bit-reproducible); the three-seed numbers of the
  paper are computed from the released checkpoints.
- `stage1_encoder/evaluate.py` and `stage1_encoder/export_hf.py`, which the
  pre-training job script calls, are now part of the repository. The
  diagnostics are computed on every spectrum in a fixed order (the previous
  version reused the shuffled training loader, which dropped the last
  incomplete batch).

### Documentation corrected

- **Encoder split.** BeMAE is pre-trained, without labels, on the Hα
  spectra of all stars except eight held-out stars; it does not use the
  70/15/15 star-level partition shipped with the dataset. The latent-space
  diagnostics are computed on these eight stars (Recall@10) and on the 22
  stars observed in both emission and absorption (k-NN accuracy).
- **EWForecast population.** The task is scored on every star with enough
  epochs for one context window (621 stars for a context of 5), not on the
  194 stars with at least 30 epochs as the submitted manuscript stated.
  The reported numbers are unchanged. The results restricted to the
  well-sampled stars (193 with at least 30 scored epochs, 76 % of the
  test pairs) are given by `downstream/t3_well_sampled_stratification.py`
  and in an appendix table: every ratio is lower there (PCA(10)+Ridge
  0.918, TimesFM-2.0 0.919, Chronos-Bolt-base 0.938).
- **Encoder size.** ≈795 k parameters (≈912 k with the training-time
  decoder).
- **Hardware.** All runs used one NVIDIA RTX 2080 Ti (RTX A6000 for the
  Chronos-Bolt / TimesFM inference); ≈20 GPU-hours in total.
- **Dataset documentation.** No quality-based rejection is applied at
  corpus level (all 339 115 harvested rows are released, with quality
  flags); the released rows have 35 columns.
- **Figures.** `fig_pca_scree.pdf` and `fig_t3_horizon.pdf` are regenerated with embedded TrueType fonts (`pdf.fonttype = 42`) instead of Type 3 fonts; the content is unchanged.
- **BeSS operator.** The laboratory operating BeSS is named LIRA (Laboratoire
  d'instrumentation et de recherche en astrophysique, Observatoire de Paris-PSL),
  which absorbed LESIA on 1 January 2025; the dataset card, datasheet, Croissant
  file and export script are updated accordingly, and the dataset card quotes the
  acknowledgement sentence requested by BeSS.

### Pipeline scripts

- `scripts/cluster/*.slurm` resolve the repository root correctly, share
  their settings through `_common.sh`, no longer call scripts that are not
  part of the repository, and write their outputs where the analysis
  scripts read them (`reference_results/`).
- `scripts/reproduce_paper_seed42.sh` gains `ALL_SEEDS=1` (three seeds and
  the headline ratio), `EXTRA_TSFM=1` (additional forecasters) and
  `MODEL_DIR` (local copy of the model repository).

### Figures

- Fig. 1: the wavelength axis covers the Hα ± 50 Å window actually used.
- Fig. 2 (quality profile): the annotation "(benchmark threshold)" next to
  the 30-spectra line was removed (no task uses a 30-epoch threshold) and the
  SNR line is labelled as the gate applied to scored targets only;
  `fig2_dataset_quality.py` accepts `BESS_META_PATH`.
- Horizon figure (`stats/figure_t3_horizon.py`): the legend entry
  "BeMAE+PCA(10)" was renamed "PCA(10)+Ridge" (this baseline is a ridge on
  the PCA of the raw flux and does not use BeMAE).

## Known issues of the v1.0 data files

Found after peer review; the data files are unchanged and the fixes are
scheduled for the next data version.

- For 2 311 rows (0.7 %: 2 240 MUSICOS orders, 69 VLT/FLAMES spectra,
  2 others) the `wavelength` array is in nm instead of Å while
  `lambda_min` / `lambda_max` are in Å; one further row has inconsistent
  wavelength metadata. The rows are listed in
  `dataset/known_issues_v1.0.csv`. None of them enters a scored target.
- The 69 VLT/FLAMES rows are labelled `observer_type = "amateur"`; they are
  professional observations.

## Submitted version (May 2026)

Release that accompanied the submission.
