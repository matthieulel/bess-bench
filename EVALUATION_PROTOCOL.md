# BESS-Bench — Evaluation Protocol (v1.0)

This document is the **authoritative specification** for every number
reported on BESS-Bench. Any reproduction of results must follow it
*to the letter*. Numbers obtained with a different protocol are not
comparable.

The protocol is deliberately conservative: the dataset defines exactly
**three tasks** — SpecProbe, LineTransfer, and EWForecast — each
with a single primary metric and fixed reference baselines reported
under a unified statistical protocol (three encoder seeds, bootstrap CI₉₅).

---

## 0. Global rules (apply to every task)

1. **Data source.** Load the dataset exclusively from the released
   Hugging Face snapshot
   (`datasets.load_dataset("matthieulel/bess-bench")`,
   `revision=<frozen tag>`). Do not mix in private scrapes of BeSS.
2. **Splits.** Three mechanisms keep training and evaluation data apart:
   - *SpecProbe / LineTransfer*: 5-fold `GroupKFold` on the star
     identifier, over all scored stars (no star on both sides of a fold).
   - *EWForecast*: a per-star temporal cutoff at `MJD = 59215.0`
     (train < cutoff ≤ test).
   - *Leaderboard partition*: `splits.csv` bundled with the release
     (`split ∈ {train, validation, test}`, 1 024 / 219 / 225 stars,
     70 / 15 / 15 %, assigned deterministically from an MD5 hash of the
     star name). Submissions that **train on labels** must fit on
     `train`, select on `validation` and report on `test`. The v1.0
     reference baselines are probes evaluated by cross-validation and
     therefore do not use this partition.
3. **Test contamination.** No label of a held-out fold, of a test star
   or of a post-cutoff epoch may be used for training, hyper-parameter
   selection or early stopping. Label-free pre-training on spectra of
   evaluation stars (the *transductive* setting) is allowed but **must
   be declared**. This is the setting of the v1.0 reference encoder:
   BeMAE is pre-trained without labels on the Hα spectra of all stars
   except eight held-out stars used for early stopping (GAM CAS,
   PLEIONE, ZET TAU, V442 AND, PI AQR, DEL SCO, 28 CYG, OME CMA), so its
   SpecProbe scores measure the quality of the representation on stars
   whose unlabelled spectra were seen during pre-training.
4. **Seed.** Every randomised component (train sampler, init, dropout,
   bootstrap) must use an explicit integer seed. Star-grouped folds and
   the leaderboard partition are deterministic. Multi-seed runs use
   `{42, 123, 456}` unless stated otherwise.
5. **Reporting.** Every reported result must serialise to a JSON file
   with the layout of the corresponding file in `reference_results/`
   (the layout read by `stats/aggregate_results.py`). Missing tasks are
   acceptable; silently changing a metric definition is not.
6. **Compute budget disclosure.** Report GPU-hours, GPU model, peak
   VRAM, and wall-clock time per task. We do not cap compute but we
   make it visible.

---

## 1. SpecProbe — Hα single-line spectral-feature regression (Ridge probe)

### 1.1. Task
Given a frozen embedding `z ∈ ℝᵈ` of a single spectrum, predict six
spectral features of the Hα line computed by
`downstream/compute_spectral_features.py`:

| Feature          | Unit | Coverage  | Notes                                   |
|------------------|------|-----------|-----------------------------------------|
| `ew`             | Å    | 26 858 sp | Pseudo-equivalent width                 |
| `fwhm`           | Å    | 26 858 sp | Full width at half maximum              |
| `central_depth`  | —    | 26 858 sp | 1 − F_min / F_ref: F_min = min flux over the 7 central bins (±2.4 Å); F_ref = strongest emission peak within ±20 Å if one exists (height ≥ 1.05), else the continuum (1) |
| `delta_v`        | km/s | 10 804 sp | Doppler shift, double-peaked only       |
| `vr_ratio`       | —    | 10 804 sp | V/R peak ratio, double-peaked only      |
| `peak_intensity` | —    | 26 858 sp | Max(normalised flux) within ±15 Å (I_peak / I_c) |

### 1.2. Protocol
- **Probe.** `sklearn.linear_model.RidgeCV` (α grid: six decade-spaced
  values in `[1e-2, 1e3]`) fitted on standardised `z`, inside each
  training fold. No feature engineering on top of `z`.
- **Evaluation.** 5-fold `GroupKFold` on `star_id` (no star shared
  between train and test of any fold).
- **Multi-seed.** Train the encoder with seeds `{42, 123, 456}` and
  report `R²_mean ± R²_std` over seeds (the encoder seed is the unit
  of variability; CV folds are pooled within a seed).
- **Primary metric.** `R²_mean` per feature. Secondary: `R²_std` over
  seeds.
- **Reference baselines** (bundled, same probe and folds):
  - `PCA(10)`, `PCA(32)` and `PCA(50)` of the normalised flux
    (PCA fitted on the scored Hα slice, without labels).
  - the raw 128-bin normalised flux.
- **Headline aggregated test.** Bootstrap CI₉₅ of the ratio
  `R²(z) / R²(PCA(10))` aggregated over the three shape features
  (`fwhm`, `central_depth`, `delta_v`) by seed-level resampling
  (n = 3 seeds, 10 000 replicates, 2.5/97.5 percentiles). Aggregating
  rather than reporting three independent comparisons controls the
  family-wise error rate (≈14 % at α = 0.05).
- **Per-feature significance rule.** A submission *strictly beats*
  PCA(10) on feature `f` when the Gaussian 95 % CIs `μ ± 1.96·σ/√k`
  (k=5 folds) are **disjoint** (see
  `stats/bootstrap_ci.py::t1_probe_ci`). Overlapping CIs count as a
  tie.

### 1.3. Reporting
- One `R²_mean ± CI95` per feature per method.
- Emit `ci_results.json["specprobe"]` as defined by `stats/bootstrap_ci.py`.

---

## 2. LineTransfer — Cross-line Hβ → Hα probing

### 2.1. Task
For every same-night `(star, MJD)` pair that covers both Hα and Hβ within
±50 Å (n = 2 525 pairs over 1 090 stars), predict the six Hα features
(§1.1) from the **Hβ window flux** alone — testing how much of the Hα
line information is carried by another Balmer line.

### 2.2. Protocol
- **Input.** 128-bin normalised flux on the Hβ ±50 Å window emitted by
  `downstream/compute_hbeta_features.py`.
- **Probe.** Standardisation + `sklearn.linear_model.RidgeCV` (same α
  grid as SpecProbe) from a representation of the Hβ window to each Hα
  target (one probe per target).
- **Evaluation.** 5-fold `GroupKFold` on `star_id`; `R²_mean ± std`
  across folds (the linear baselines are deterministic, there is no
  seed).
- **Primary metric.** `R²_mean(ew)` (most physically meaningful: Hα
  and Hβ EWs are correlated through the underlying disk emission;
  v1.0 reaches 0.66 with PCA(10), 0.66 with PCA(50) and 0.71 with the
  raw Hβ flux). Secondary: per-feature `R²_mean ± std` for
  `{fwhm, central_depth, delta_v, vr_ratio, peak_intensity}`
  (line-shape features do not transfer reliably in v1.0; `delta_v` and
  `vr_ratio` are defined on fewer than 200 pairs).
- **Reference baselines.** PCA(10), PCA(50) and raw flux of the Hβ
  window, same protocol. The v1.0 reference encoder (BeMAE) is trained
  on the Hα window only and is not evaluated on this task.

### 2.3. Reporting
Emit a JSON file with the layout of
`reference_results/downstream/probe_t2_hbeta_to_halpha.json`.

---

## 3. EWForecast — One-step EW(Hα) forecasting

This is the **headline temporal protocol** of the benchmark.

### 3.1. Task
For every star of the scored Hα slice (SNR ≥ 10 gate), order the epochs
in time, form rolling context windows of the `L ∈ {5, 10}` past epochs
ending at `t_k` and predict `EW(t_{k+1})` at the next available epoch.
The reference implementation enforces a temporal split at
`MJD = 59215.0` (2021-01-01; train < cutoff, test ≥ cutoff) so that no
test epoch is ever used in PCA fitting or Ridge fitting. The elapsed
time `t_{k+1} − t_k` is **not** a model input: the task is
next-observation forecasting, not fixed-horizon forecasting.

**Test population.** A star contributes as soon as it has enough epochs
for one context window; there is no minimum of 30 epochs.

| Setting | Test pairs | Stars |
|---|---:|---:|
| trained baselines, `L = 5` | 7 313 | 621 |
| trained baselines, `L = 10` | 6 664 | 432 |
| zero-shot forecasters (context of 1 to 64 past epochs) | 8 118 | 846 |

The 193 stars with at least 30 scored epochs account for 76 % of the
`L = 5` test pairs; `downstream/t3_well_sampled_stratification.py`
reports the ratios on these stars and on the others separately.

### 3.2. Protocol
- **Persistence baseline** (canonical):
  `ŷ_{k+1} = ew(t_k)`. Used as the denominator of the primary metric.
- **Reference learned baseline.** `PCA+Ridge` over context windows —
  `downstream/t3_pca_ridge_temporal_baseline.py`. Four configurations:
  `{PCA(10), PCA(50)} × {ctx=5, ctx=10}`. Best in-domain
  configuration: PCA(10)·ctx=5 → ratio 0.942 (the linear head on frozen
  Chronos-Bolt-base reaches 0.932).
- **Reference zero-shot baselines.** Chronos-Bolt (small, base) and
  TimesFM-2.0-500 M — `downstream/t3_ts_fm_baselines.py`; Moirai,
  Moirai-MoE and Time-MoE — `downstream/t3_moirai_zeroshot.py`,
  `t3_moiraimoe_zeroshot.py`, `t3_timemoe_zeroshot.py`. Zero-shot
  scores are lower bounds on what trained or fine-tuned models reach.
- **Other reference baselines.** Ridge on the BeMAE embeddings of the
  context (`t3_z_ridge_temporal_baseline.py`), a trained linear head on
  frozen Chronos-Bolt forecasts (`t3_chronos_trained_head.py`) and
  time-aware ridge variants (`t3_time_aware_ridge.py`).
- **Negative reference (not a baseline to beat).** The causal
  Transformer head on BeMAE embeddings (`stage2_transformer_head/`,
  ratio 1.17 ± 0.05 over three seeds) is scored with its own harness
  (6 495 pairs from 479 stars) and its early stopping monitors the
  post-cutoff loss. It therefore does not satisfy rule 0.3; this only
  favours it, and it still does not beat persistence.
- **Primary metric.**
  `ratio_mae = mae_model / mae_persistence`
  on the pooled test pairs (all stars, all horizons), with the
  persistence error computed on the same pairs as the model.
  Smaller is better; `ratio_mae < 1` is the minimum bar to claim any
  improvement over persistence.
- **Bootstrap CI95.** Non-parametric bootstrap on the per-epoch absolute
  errors (n = 10 000 resamples, 2.5/97.5 percentiles); see
  `stats/bootstrap_ci_t3.py`.
- **Claim rule.** A published claim of beating persistence requires
  the bootstrap CI95 to lie **strictly below 1** (`upper < 1`). A point estimate below 1 with a CI crossing 1 is a
  tie.

### 3.3. Secondary diagnostics
Emit `by_horizon` R² and MAE for the buckets
`{<7d, 7–30d, 30–90d, 90–365d, >365d}` (matches
`t3_pca_ridge_temporal_baseline.json::by_horizon`). v1.0 finding:
PCA(10)+Ridge (ctx=5) improves on persistence in every bucket up to
365 d (ratios 0.91–0.96) and not beyond (ratio 1.01).

### 3.4. Reporting
One row per configuration plus a bootstrap summary, in the layout read
by `stats/aggregate_results.py::table_t3_baselines`
(`summary.json["t3_baselines"]`).

---

## 4. What is **not** part of the benchmark (and why)

- **Wilcoxon / McNemar / permutation tests.** Paired tests between
  methods are not part of the v1.0 protocol; they are planned for a
  future version.
- **Change-point detection.** Out of scope for v1.0: stable expert
  annotations of outburst / disk-loss events are not yet released.
- **Phase classification.** A fourth task (phase classification on
  Hα morphological classes) is **deferred to v1.1**: labels are not
  released and no v1.0 baseline is scored.
- **Zero-shot / in-context evaluation of foundation LMs.** Out of
  scope for v1.0 because SpecProbe and LineTransfer are pure probing
  tasks; this can change in later revisions.
- **Pre-training metrics** (reconstruction loss, contrastive score).
  Pre-training is not a task; v1.0 reports SpecProbe, LineTransfer and
  EWForecast.

---

## 5. Versioning

This document is `v1.0`. Any breaking change to a metric, split, or
significance rule bumps the major version and invalidates prior
results. Patch versions (`v1.0.x`) may clarify wording but never
change numbers.

**Change with respect to the submitted manuscript.** The reference
implementation of SpecProbe and LineTransfer used spectrum-level folds
when the paper was submitted, although this document already specified
star-grouped folds. The code now follows this document; the previous
outputs are kept in `reference_results/submission_ungrouped/` and the
change is described in `CHANGELOG.md`.
