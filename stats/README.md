# stats/ — Statistical aggregation and confidence intervals

Scripts that aggregate the raw downstream JSON outputs, compute confidence
intervals, and emit the tables and figures used in the paper.

## Pipeline

```bash
# 1. Multi-seed aggregation -> reference_results/stats/*.json
python stats/aggregate_t1_multiseed.py   # SpecProbe: mean ± std over seeds, headline ratio (Eq. 1)
python stats/aggregate_ablations.py      # ablation grid: headline ratio per variant, table_ablations.tex

# 2. Confidence intervals
python stats/bootstrap_ci.py             # SpecProbe: Gaussian CI95 over the 5 folds (seed 42)
python stats/bootstrap_ci_t3.py          # EWForecast: bootstrap CI95 (n = 10 000)

# 3. LaTeX tables -> paper/tables/*.tex and paper/tables/summary.json
python stats/aggregate_results.py

# 4. Figures
python stats/figure_t3_horizon.py        # MAE ratio stratified by forecast horizon
```

`paper/figure_scripts/` holds the scripts of the other figures.

## Key output files

| Path | Content |
|---|---|
| `reference_results/stats/t1_multiseed_summary.json` | SpecProbe per-feature R² over encoder seeds {42, 123, 456} and the headline ratio |
| `reference_results/stats/ablations_summary.json`    | headline ratio for each encoder ablation (mask ratio, depth, width, patch size) |
| `reference_results/stats/t3_bootstrap_cis.json`     | EWForecast bootstrap CI95 for the PCA(K)+Ridge configurations and the TS-FMs |
| `reference_results/stats/coverage_audit.json`       | per-line wavelength coverage of the corpus |
| `paper/tables/*.tex`, `paper/tables/summary.json`   | LaTeX tables of the paper and the numbers they contain |

## Methodological choices

### SpecProbe — variability across seeds
Each encoder seed reports R² per feature averaged over 5 `GroupKFold`
folds (`star_id` as group). The paper reports the mean ± sample standard
deviation (ddof = 1) over the three seeds. The linear baselines (PCA,
raw flux) do not depend on the seed.

### SpecProbe — headline aggregated ratio
The headline statistic is `R²(z) / R²(PCA(10))`, with R² averaged over
the three shape features (`fwhm`, `central_depth`, `delta_v`); its CI95
comes from seed-level resampling (n = 3 seeds, 10 000 replicates,
2.5/97.5 percentiles). v1.0 result: **7.77 [7.26, 8.06]**. With three
seeds this interval describes the spread across seeds; it is not a
calibrated confidence interval.

### SpecProbe — per-feature Gaussian CI95
`mean ± 1.96 · std / √k` over the k = 5 folds of the seed-42 run
(`bootstrap_ci.py`), reported in the appendix table.

### EWForecast — bootstrap CI95
Non-parametric bootstrap on the per-prediction absolute errors
(n = 10 000 resamples of the test pairs). A baseline beats persistence
when the CI95 of the MAE ratio lies entirely below 1. v1.0 best in-domain result:
PCA(10)+Ridge, ctx=5 → MAE ratio 0.942, CI95 [0.927, 0.957] (the linear
head on frozen Chronos-Bolt-base reaches 0.932, CI95 [0.918, 0.947]).

### Not performed
- **Paired Wilcoxon / McNemar / permutation tests.** The EWForecast
  outputs store the per-prediction `y_true / y_pred` arrays, but the
  claim rule of the paper is built on the bootstrap CI95; paired
  non-parametric tests are left for a later version.
- **Phase classification.** Deferred (labels not finalised; see
  `EVALUATION_PROTOCOL.md` §4).

## Reproducibility

All scripts are deterministic (random seeds fixed in each script).

## Dataset release helpers

`generate_star_list.py` (star list and leaderboard partition),
`generate_croissant.py` (Croissant metadata), `extract_metadata_slim.py`,
`coverage_audit.py`, `figure_quality_profile.py`, `figure_teaser.py` and
`build_sample.py` operate on a local copy of the full dataset and were
used to prepare the release.
