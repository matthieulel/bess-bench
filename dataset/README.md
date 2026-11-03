---
license: cc-by-4.0
language:
- en
pretty_name: BESS-Bench
size_categories:
- 100K<n<1M
task_categories:
- feature-extraction
- time-series-forecasting
- other
task_ids:
- multivariate-time-series-forecasting
tags:
- astronomy
- astrophysics
- spectroscopy
- be-stars
- stellar-spectra
- time-series
- representation-learning
- benchmark
- citizen-science
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# BESS-Bench: Long-baseline, mixed-quality Be-star spectra for ML time-series research

> **339 115 spectral rows · 37 624 physical observations · 1 468 Be stars · 35 years · 91.6 % of observations from amateur observers**
> A benchmark for representation learning and temporal modelling on heterogeneous
> stellar spectroscopic time series.
>
> **Three granularity levels** (see §Dataset composition): parquet *rows*
> (atom of the file), *physical observations* (one observing session ≡
> one `(star_name, mjd)` group; an échelle observation spans a median of
> 26 orders), and *unique stars* (split unit). The amateur share is
> $91.6\,\%$ at the observation level and $81.7\,\%$ at the row level
> (échelle orders multiply professional row counts).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## TL;DR

BESS-Bench is an ML-ready, per-star split release of the BeSS community Be-star
spectral archive, together with:

- **Full-range optical spectra** (1 150 – 10 442 Å native coverage). The
  fraction of spectra that *strictly* cover each commonly studied Be-star
  line at ±50 Å is reported at two granularities — **per physical
  observation** (Hα 77.6 %, Hβ 35.7 %, He I 5876
  36.0 %, Na D 36.0 %, Hγ 32.5 %, Fe II 5169 35.6 %, Hδ 19.4 %, O I 7772 4.0 %)
  and **per parquet row** (row-level coverage used inside tasks: Hα 7.9 %,
  Hβ 0.8 %, …). Both tables are reproduced in §*Line coverage* below.
- A pre-trained Hα-window reference encoder, **BeMAE** (≈795 K parameters,
  three seeds), released separately as
  [`matthieulel/bemae-halpha-v1`](https://huggingface.co/matthieulel/bemae-halpha-v1).
- **Three benchmarked downstream tasks** (**SpecProbe** feature regression,
  **LineTransfer** Hβ → Hα generalisation, **EWForecast** short-horizon
  EW(Hα) forecasting), with fixed protocols and bootstrap / multi-seed
  confidence intervals. The three tasks focus on Hα and Hβ because
  these are the only two lines with row-level coverage large enough
  to support meaningful confidence intervals in v1.0.
- A task-dependent ranking: the learned encoder dominates every linear
  representation on line-shape features (SpecProbe), but does not beat
  persistence on EWForecast, where a simple PCA+Ridge baseline does
  (MAE ratio **0.942**, bootstrap CI95 [0.927, 0.957]) and is ahead of
  nine zero-shot time-series foundation-model checkpoints; a linear head
  trained on top of frozen Chronos-Bolt-base reaches 0.932.

Full methodology, datasheet and reproducibility instructions are in the
accompanying paper and in [DATASHEET.md](./DATASHEET.md). Result-level
JSON artefacts (per-task summaries, bootstrap CIs, coverage audits)
live in the companion code repository; this card quotes their content
for self-containedness.

## Quick start

```python
from datasets import load_dataset

# Full dataset (10 GB, 21 shards)
ds = load_dataset("matthieulel/bess-bench", split="train")

# Streaming access (no full download)
ds_stream = load_dataset("matthieulel/bess-bench", split="train", streaming=True)

# A small stratified sample (1,976 spectra, 150 stars, ~70 MB) for quick inspection
sample = load_dataset("matthieulel/bess-bench-sample", split="train")

# Apply the official per-star split
import pandas as pd
splits = pd.read_csv("hf://datasets/matthieulel/bess-bench/splits.csv")
train_stars = set(splits.query("split == 'train'").star_name)
train = ds.filter(lambda x: x["star_name"] in train_stars, num_proc=8)
```

Reproduction of every paper table from frozen checkpoints is documented
in `REPRODUCE.md` of the companion code repository.

## Dataset composition

### Three granularity levels

BESS-Bench is distributed as a parquet table, and we distinguish three
nested units of measurement:

| Unit                       | Count   | Definition                                      |
|----------------------------|---------|-------------------------------------------------|
| Parquet rows               | 339 115 | One row = one 1D spectrum or one échelle order  |
| **Physical observations**  |  37 624 | One observing session ≡ one `(star_name, mjd)`  |
| Unique stars (split unit)  |   1 468 | Per-star split, no leakage                      |

The ~9× inflation from physical observations to parquet rows comes from
**9 917 échelle sessions** each stored as 19–40 individual orders
(median 26 orders / session) — an intentional choice that preserves the
native resolution (R ≈ 10 000 – 17 000). The 27 707 remaining
observations are single-range spectra (one row = one observation).

| Stream          | Rows    | Physical obs. | Unique stars | Median R     |
|-----------------|---------|---------------|--------------|--------------|
| Single-range    |  62 476 |  27 707       |  1 354       | 9 000        |
| Échelle orders  | 276 639 |   9 917       |    915       | 11 000       |
| **Total**       | 339 115 |  37 624       |  1 468       | —            |

### Summary metrics

| Metric                         | Value                               |
|--------------------------------|-------------------------------------|
| Parquet rows                   | 339 115                             |
| Physical observations          |  37 624                             |
| Unique stars                   |   1 468                             |
| Temporal coverage              | 1990-03 → 2025-12 (35 years)        |
| Wavelength coverage            | 1 150 – 10 442 Å                    |
| Median SNR                     | 189                                 |
| Amateur fraction (rows)        | 81.7 %                              |
| Amateur fraction (obs.)        | 91.6 %                              |
| Spectrographs / telescopes     | 26 / 62                             |
| Licence                        | CC-BY-4.0                        |

Filter échelle orders with the `is_echelle_order` flag. v1.0 downstream
tasks group by `(star_name, mjd, instrument_setup)` so that an échelle
observation contributes once, not 26 times.

### Wavelength frame convention

The `wavelength` column is shipped **as in the BeSS FITS** (no shift
applied), so users can reproduce any rest frame they wish. For ≈ 97 %
of spectra the released grid is therefore **topocentric**. To reach
the heliocentric frame, apply

    λ_helio = λ_obs · (1 − bss_rqvh / c)         (c = 299 792.458 km/s)

`helio_velocity` (= FITS `BSS_VHEL`) is the correction *already
applied* by the observer (0 for 97 % of spectra) and `bss_rqvh`
(= FITS `BSS_RQVH`) is the residual correction *still to apply*
(populated for 99.8 % of spectra, ±20 km s⁻¹, σ ≈ 10 km s⁻¹).

The bundled BESS-Bench baselines (encoder pre-training, SpecProbe/LineTransfer/EWForecast)
**apply this correction internally** before the 128-bin Hα/Hβ
crop so that features and embeddings live in a common heliocentric
frame compatible with external surveys (SDSS, LAMOST, APOGEE).

### Line coverage — two complementary tables

BESS-Bench ships the native wavelength coverage of each observation (not
a cropped Hα window). Two coverage tables are computed on the v1.0
snapshot (audit script and full JSON in the companion code repository).

**Table A — coverage per physical observation** *(astrophysical view:
of the 37 624 observing sessions, which contain this line at ±50 Å?
Échelle rows are aggregated via λ<sub>min</sub>=min, λ<sub>max</sub>=max
over their orders.)*

| Line        | λ₀ (Å)  | Coverage | Observations covered (out of 37 624) |
|-------------|--------:|---------:|-------------------------------------:|
| Hα          |  6562.8 | 77.63 %  | 29 207                               |
| Na D        |  5893.0 | 35.98 %  | 13 536                               |
| He I 5876   |  5876.0 | 35.97 %  | 13 532                               |
| Hβ          |  4861.3 | 35.66 %  | 13 416                               |
| Fe II 5169  |  5169.0 | 35.55 %  | 13 376                               |
| Hγ          |  4340.5 | 32.54 %  | 12 242                               |
| Hδ          |  4101.7 | 19.45 %  |  7 317                               |
| O I 7772    |  7772.0 |  4.05 %  |  1 523                               |

**Table B — coverage per parquet row** *(per-row coverage, used when
tasks iterate over individual parquet rows rather than over observing
sessions)*

| Line        | λ₀ (Å)  | Coverage | Rows covered (out of 339 115) |
|-------------|--------:|---------:|------------------------------:|
| Hα          |  6562.8 |  7.94 %  |  26 937                       |
| Na D        |  5893.0 |  3.37 %  |  11 413                       |
| He I 5876   |  5876.0 |  1.94 %  |   6 570                       |
| Hβ          |  4861.3 |  0.78 %  |   2 652                       |
| Fe II 5169  |  5169.0 |  0.71 %  |   2 421                       |
| Hγ          |  4340.5 |  0.69 %  |   2 337                       |
| Hδ          |  4101.7 |  0.66 %  |   2 246                       |
| O I 7772    |  7772.0 |  0.41 %  |   1 387                       |

The 10× gap between the two tables is explained entirely by the échelle
decomposition: each échelle row covers a narrow sub-range by
construction, so a single observing session that covers Hα contributes
26 rows in Table B but only 1 observation in Table A. Wavelength
distribution across rows: λ<sub>min</sub> p5/p50/p95 = 3 980 / 5 277 /
7 006 Å; λ<sub>max</sub> p5/p50/p95 = 4 113 / 5 413 / 7 351 Å.

v1.0 baselined tasks (SpecProbe, LineTransfer, EWForecast) use the
Hα and Hβ windows because they combine high observation-level
coverage (77.6 % / 35.7 %) with enough row-level density (26 937 /
2 652 spectra) for stable CVs. He I, Na D and Fe II 5169 each reach
~36 % observation-level coverage and are released for community
experimentation, but no v1.0 leaderboard task targets them.

### Splits

Split **by star** (not by spectrum) to prevent temporal leakage:

| Split       | Stars | Spectra  | Fraction |
|-------------|-------|----------|----------|
| train       | 1 024 | 241 057  | ≈71 %    |
| validation  | 219   | 44 383   | ≈13 %    |
| test        | 225   | 53 675   | ≈16 %    |

Assignment is deterministic via
`md5("bess_bench_v1::" + star_name)[:8] mod 100` ↦
{0–69: train, 70–84: val, 85–99: test}. The result is bundled as
`splits.csv`; the generating script is shipped in the companion code
repository for full reproducibility.

**How the splits are used.** This star-level partition is the reference
split for submissions that train on labels. The v1.0 baselines of the
paper do not need it: SpecProbe and LineTransfer use 5-fold `GroupKFold`
on `star_name` over all scored stars, EWForecast uses a per-star temporal
cutoff (MJD 59215), and the BeMAE encoder is pre-trained without labels
on all Hα spectra except those of eight held-out stars (GAM CAS, PLEIONE,
ZET TAU, V442 AND, PI AQR, DEL SCO, 28 CYG, OME CMA).

## Schema

| Column                          | Type           | Description                                  |
|---------------------------------|----------------|----------------------------------------------|
| `wavelength`                    | list[float32]  | Native wavelength grid (Å; in nm for 2 311 rows, see *Known issues*) |
| `flux`                          | list[float32]  | Calibrated flux                              |
| `flux_error`                    | list[float32]? | Flux uncertainty (null for a substantial fraction of amateur FITS; see DATASHEET §3) |
| `spectrum_id`                   | string         | MD5 (star_name, mjd, instrument_setup)       |
| `star_name`                     | string         | Canonical de-duplicated name (1 468 unique)  |
| `star_name_raw`                 | string         | Original BeSS object name                    |
| `ra`, `dec`                     | float64        | J2000 (degrees)                              |
| `lambda_min`, `lambda_max`      | float64        | Spectral coverage (Å)                        |
| `n_pixels`                      | int32          | Number of spectral pixels                    |
| `spectral_resolution`           | float64        | Theoretical R                                |
| `spectral_resolution_measured`  | float64?       | Measured R (null when unavailable)           |
| `snr`                           | float64?       | DER_SNR (null for 29 corrupted spectra)      |
| `snr_continuum`                 | float64        | Continuum-window SNR                         |
| `snr_quality`                   | string         | excellent/good/medium/low/very_low/unknown   |
| `observation_date`              | string         | ISO-8601 date                                |
| `mjd`                           | float64        | Modified Julian Date                         |
| `exposure_time`                 | float64        | Seconds                                      |
| `temporal_quality`              | string         | good / no_exptime / very_long                |
| `spectrograph`                  | string         | Normalised                                   |
| `telescope`                     | string         | Normalised                                   |
| `detector`                      | string         | Normalised                                   |
| `instrument_setup`              | string         | Canonical (spectrograph, telescope) pair     |
| `instrument_confidence`         | string         | high / medium / low                          |
| `observer_type`                 | string         | amateur / professional / unknown             |
| `site_latitude`                 | float64?       | Degrees, rounded to 0.1° (≈11 km)              |
| `site_longitude`                | float64?       | Degrees, rounded to 0.1° (≈11 km)              |
| `site_elevation`                | float64?       | Metres, rounded to 100 m                     |
| `helio_velocity`                | float64?       | Heliocentric correction *already applied* by the observer (= FITS `BSS_VHEL`, km s⁻¹). 0 for ≈97 % of spectra. |
| `bss_rqvh`                      | float64?       | Residual heliocentric correction *to apply* to reach the heliocentric frame (= FITS `BSS_RQVH`, km s⁻¹). ±20 km s⁻¹, 1σ≈10 km s⁻¹, populated for 99.8 % of spectra. |
| `telluric_corrected`            | string         | Raw FITS header string (free-form, often `None`) |
| `normalized`                    | string         | Raw FITS header string (free-form, often `None`) |
| `is_echelle_order`              | bool           | True when the row is one order of an echelle |
| `fits_format`                   | string         | Original FITS flavour                        |

## Benchmark protocol

BESS-Bench v1.0 releases **three quantitative tasks** (SpecProbe,
LineTransfer, EWForecast). Every task reports reproducible mean ± std
and/or bootstrap CI95 over seeds {42, 123, 456} or 10 000 bootstrap
resamples, under the rule that a learned-model claim of improvement
requires the corresponding CI to exclude the baseline value.

### SpecProbe — Hα single-line spectral-feature regression (Ridge probe, 3 encoder seeds)

Frozen 128-D embeddings from the released Hα-window encoder → Ridge
regression on six Hα spectral features. **5-fold GroupKFold on
`star_name`** (no star on both sides of a fold) over the 26 858 scored
Hα spectra of 1 428 stars (Δv / vr_ratio are defined on the 10 804
cleanly double-peaked profiles), repeated for encoder seeds
{42, 123, 456}. Baselines: the same probe on PCA(10/32/50) and on the raw
128-bin normalised flux. R², mean ± std over the three seeds (the
linear baselines are deterministic):

| feature | z (128-D, BeMAE) | PCA(10) | PCA(50) | raw flux (128-D) |
|---|---|---|---|---|
| `ew` | 0.993 ± 0.002 | 1.000 | 1.000 | 1.000 |
| `peak_intensity` | 0.993 ± 0.003 | 0.950 | 0.947 | 0.948 |
| `fwhm` | **0.508 ± 0.037** | 0.094 | 0.125 | 0.121 |
| `central_depth` | **0.879 ± 0.023** | -0.001 | -0.028 | -0.016 |
| `delta_v` | **0.412 ± 0.056** | 0.139 | 0.192 | 0.188 |
| `vr_ratio` | 0.129 ± 0.068 | 0.117 | 0.030 | 0.063 |

Aggregated ratio of mean R² over `fwhm`, `central_depth`, `delta_v`,
z vs PCA(10): **7.77** (seed-level bootstrap CI95 [7.26, 8.06]). The
encoder adds clear value on the non-linear shape features; EW and peak
intensity are linearly decodable from the flux, and `vr_ratio` sits near
its noise floor for every representation.

> These numbers supersede those of the submitted manuscript, which had
> been computed with spectrum-level folds (stellar identity leaking
> across folds). The ranking is unchanged; absolute R² values are lower.

### LineTransfer — Cross-line probing (Hβ → Hα features)

Same Hα targets as SpecProbe; the input is the Hβ ± 50 Å window (128 bins,
locally normalised), restricted to the 2,525 same-night
(star, MJD) pairs from 1,090 stars where both lines are
observed. Ridge probe on a linear representation of Hβ, 5-fold
GroupKFold on `star_name`; R², mean ± std across folds:

| Hα target | pairs | Hβ PCA(10) | Hβ PCA(50) | Hβ raw flux |
|---|---|---|---|---|
| `ew` | 2,525 | 0.66 ± 0.06 | 0.66 ± 0.06 | 0.71 ± 0.07 |
| `peak_intensity` | 2,525 | 0.28 ± 0.19 | 0.38 ± 0.10 | 0.17 ± 0.56 |
| `fwhm` | 2,508 | 0.00 ± 0.09 | 0.05 ± 0.02 | 0.00 ± 0.04 |
| `central_depth` | 2,525 | -0.01 ± 0.07 | 0.03 ± 0.07 | 0.00 ± 0.04 |
| `delta_v` | 190 | -0.08 ± 0.22 | -0.11 ± 0.15 | 0.16 ± 0.14 |
| `vr_ratio` | 154 | -0.04 ± 0.03 | -0.02 ± 0.03 | -0.07 ± 0.05 |

Equivalent width transfers (R² ≈ 0.66–0.71); line-shape parameters do
not. `delta_v` and `vr_ratio` are defined on fewer than 200 pairs and are
inconclusive. BeMAE is an Hα-only encoder and is not evaluated here.

### EWForecast — Short-horizon EW(Hα) forecasting (bootstrap CI95)

One-step-ahead forecast of EW(Hα) at the next observation epoch given
the past epochs of the same star. Primary metric:
`ratio_mae = MAE(model) / MAE(persistence)` on the same test pairs.
Split: per-star temporal cutoff at MJD = 59215 (2021-01-01); the PCA
basis and the ridge heads are fitted on pre-cutoff epochs only. A model
**beats persistence when the bootstrap CI95 of the ratio (10 000
resamples of the test pairs) lies strictly below 1**.

Test populations: 7,313 pairs from 621 stars for ctx=5
(6 664 pairs from 432 stars for ctx=10); zero-shot forecasters need no
minimum context and are scored on 8 118 pairs from 846 stars. The
elapsed time to the target is not a model input (next-observation
forecasting).

| Method | MAE ratio | CI95 | beats persistence? |
|---|---:|---|:---:|
| PCA(10)+Ridge, ctx=5 | 0.942 | [0.927, 0.957] | **yes** |
| PCA(10)+Ridge, ctx=10 | 0.956 | [0.937, 0.976] | **yes** |
| PCA(50)+Ridge, ctx=5 | 0.983 | [0.966, 1.002] | inconclusive |
| PCA(50)+Ridge, ctx=10 | 1.040 | [1.016, 1.065] | no |
| BeMAE z(128)+Ridge, ctx=5 | 1.003 | [0.982, 1.026] | inconclusive |
| BeMAE z(128)+Ridge, ctx=10 | 1.014 | [0.991, 1.038] | inconclusive |
| Chronos-Bolt-small (48 M), zero-shot | 0.981 | [0.966, 0.997] | **yes** |
| Chronos-Bolt-base (205 M), zero-shot | 0.968 | [0.955, 0.980] | **yes** |
| TimesFM-2.0 (500 M), zero-shot | 0.969 | [0.949, 0.989] | **yes** |
| Moirai-1.1-R-base (91 M), zero-shot | 1.021 | [1.002, 1.041] | no |
| Moirai-1.1-R-large (311 M), zero-shot | 0.992 | [0.975, 1.008] | inconclusive |
| Moirai-MoE-1.0-R-small (117 M), zero-shot | 1.004 | [0.985, 1.024] | inconclusive |
| Moirai-MoE-1.0-R-base (935 M), zero-shot | 1.061 | [1.034, 1.088] | no |
| Time-MoE-50M (113 M), zero-shot | 0.988 | [0.970, 1.006] | inconclusive |
| Time-MoE-200M (453 M), zero-shot | 0.986 | [0.968, 1.004] | inconclusive |
| Chronos-Bolt-base + trained linear head (ctx=5 pairs) | 0.932 | [0.918, 0.947] | **yes** |
| — control: ridge on the 5-epoch EW context | 0.945 | [0.931, 0.959] | **yes** |
| — Chronos-Bolt-base zero-shot on the same pairs | 0.955 | [0.941, 0.970] | **yes** |
| Causal Transformer on BeMAE embeddings (3 seeds, own harness) | 1.17 ± 0.05 | — | no |

Zero-shot numbers are lower bounds on what trained or fine-tuned
foundation models could reach. Median per-spectrum EW measurement
uncertainty (Vollmann & Eversberg 2006 applied to the shipped DER_SNR):
≈ 0.44 Å over the stars with ≥ 30 epochs (0.46 Å over the scored slice),
against a persistence MAE of 1.47 Å and a best MAE reduction of ≈ 0.09 Å:
no current baseline is scientifically decisive yet.

## Known issues in v1.0

The v1.0 data files are kept byte-identical to the version that was
peer-reviewed. Two issues were found afterwards and will be fixed in the
next data version:

- **Wavelength unit.** For 2 311 rows (0.7 %: 2 240 MUSICOS orders,
  69 VLT/FLAMES spectra, 2 others) the `wavelength` array is stored in
  **nm** instead of Å, while `lambda_min` / `lambda_max` are in Å; one
  further row has inconsistent wavelength metadata. The affected
  `spectrum_id`s are listed in [`known_issues_v1.0.csv`](./known_issues_v1.0.csv).
  Work-around: `if lambda_max / max(wavelength) > 5: wavelength *= 10`.
  None of these rows enters a scored target.
- **Observer type.** The 69 VLT/FLAMES rows (NGC 2004 members) are
  labelled `observer_type = "amateur"`; they are professional
  observations.

## Citation

```bibtex
@inproceedings{bess_bench_2026,
  title     = {BESS-Bench: Benchmarking Spectral Representations for Be-Star Variability},
  author    = {Le Lain, Matthieu and Lefèvre, Sébastien},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://huggingface.co/datasets/matthieulel/bess-bench}
}
```

The frozen v1.0 snapshot of the data files is archived on Zenodo under the DOI
[10.5281/zenodo.23808620](https://doi.org/10.5281/zenodo.23808620) (the files are byte-identical to this
repository). To cite the data themselves:

```bibtex
@dataset{bess_bench_data_v1,
  title     = {BESS-Bench v1.0: Be-star H-alpha spectra from the BeSS database},
  author    = {Le Lain, Matthieu and Lefèvre, Sébastien},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.23808620},
  url       = {https://doi.org/10.5281/zenodo.23808620}
}
```

When using BESS-Bench, please also cite the underlying BeSS database:

```bibtex
@article{neiner2011bess,
  title   = {The BeSS database},
  author  = {Neiner, C. and others},
  journal = {Astronomical Journal},
  volume  = {142},
  pages   = {149},
  year    = {2011},
  doi     = {10.1088/0004-6256/142/5/149}
}
```

## Licence

CC-BY-4.0. Attribution must credit the BeSS database (Neiner et al.
2011) and the BeSS consortium (LIRA / Observatoire de Paris).
Commercial use is permitted under the same attribution conditions.

## Acknowledgements

We thank the BeSS community of amateur and professional observers for
35 years of sustained spectroscopic monitoring. BESS-Bench packages
and structures these public observations for ML use, and credits the
original observers via the BeSS database.

Publications that use BESS-Bench should carry the acknowledgement
requested by BeSS: "This work has made use of the BeSS database,
operated at LIRA, Observatoire de Paris, France: http://basebe.obspm.fr".
