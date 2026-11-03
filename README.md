# BESS-Bench

> A benchmark for spectral representation learning and temporal forecasting on
> 35 years of collaborative Be-star spectroscopy.

Code of the paper **"BESS-Bench: Benchmarking Spectral Representations for
Be-Star Variability"** (NeurIPS 2026, Evaluations & Datasets Track).

| Artefact | Location |
|---|---|
| Dataset (339 115 spectra, 10 GB) | <https://huggingface.co/datasets/matthieulel/bess-bench> |
| Frozen v1.0 snapshot of the data (Zenodo, citable DOI) | <https://doi.org/10.5281/zenodo.23808620> |
| Inspection sample (1 976 spectra, 70 MB) | <https://huggingface.co/datasets/matthieulel/bess-bench-sample> |
| BeMAE reference encoder (three seeds) | <https://huggingface.co/matthieulel/bemae-halpha-v1> |
| Code (this repository) | <https://github.com/matthieulel/bess-bench> |

---

## What is in this repository?

| Path | Purpose |
|------|---------|
| `scripts/` | **Reproduction entry points.** `reproduce_paper_seed42.sh` (single command, from the public artefacts), `compare_with_reference.py` (diff against `reference_results/`), and `cluster/` (SLURM scripts and `run_all.sh` for the full multi-seed pipeline). |
| `stage1_encoder/` | BeMAE, the reference masked auto-encoder on the Hα window (≈795 k-parameter encoder, ≈912 k with the training-time decoder): training, diagnostics, embeddings, export. |
| `stage2_transformer_head/` | Causal Transformer head on BeMAE embeddings (EWForecast, negative result), trained and scored separately. |
| `downstream/` | The three tasks: SpecProbe (ridge probes on Hα features), LineTransfer (Hβ → Hα) and EWForecast (one-step EW(Hα) forecasting with PCA+Ridge and time-series foundation models). |
| `stats/` | Multi-seed aggregation, bootstrap confidence intervals, LaTeX tables. |
| `paper/` | Tables (`tables/`), figures (`figures/`), figure scripts (`figure_scripts/`) and validation scripts (`scripts/`) of the paper. |
| `reference_results/` | The numbers of the paper: result JSONs, embeddings of the three encoder seeds, spectral features. `submission_ungrouped/` keeps the outputs of the submitted manuscript (see `CHANGELOG.md`). |
| `dataset/` | Copy of the documentation published with the dataset: card, datasheet, Croissant metadata, star list, leaderboard partition, known issues. |
| `data_pipeline/` | Quality, clean-up and export steps that turn the harvested BeSS spectra into the released dataset. |
| `configs/ablations/` | The eight encoder ablations. |
| `EVALUATION_PROTOCOL.md` | Authoritative specification of the SpecProbe / LineTransfer / EWForecast protocol (v1.0). |
| `REPRODUCE.md` | Step-by-step reproduction guide. |
| `CHANGELOG.md` | Changes since the submitted version, and known issues. |
| `CONTRIBUTING.md` | How to contribute bug fixes and protocol clarifications. |

## Dataset access

```python
from datasets import load_dataset
ds = load_dataset("matthieulel/bess-bench", split="train")
```

Key facts:
- 339 115 parquet rows = 37 624 physical observations of 1 468 Be stars
- 35 years of observations (1990–2025)
- 91.6 % of the observations come from amateur astronomers (81.7 % of the rows)
- A star-level 70/15/15 partition (`splits.csv`) is released for
  submissions that train on labels; the reference baselines use
  star-grouped cross-validation and a temporal cutoff
  (see `EVALUATION_PROTOCOL.md`)
- Known issues of v1.0 (wavelength unit of 2 311 rows, observer type of 69
  rows) are listed in the dataset card and in `dataset/known_issues_v1.0.csv`

## Pretrained encoder

The BeMAE-Hα checkpoints of the three seeds (42, 123, 456; ≈795 k
parameters, 3 MB each) are published as a model repository. The seed-42
checkpoint is the reference one.

```python
from huggingface_hub import snapshot_download
local = snapshot_download("matthieulel/bemae-halpha-v1")
```

The model card is mirrored in `stage1_encoder/hf_export/bemae-halpha-v1/README.md`.

## Main results (v1.0)

| Task | Result |
|---|---|
| **SpecProbe** (5-fold star-grouped CV, 3 encoder seeds) | BeMAE dominates the linear representations on line-shape features: R² = 0.51 ± 0.04 (`fwhm`), 0.88 ± 0.02 (`central_depth`), 0.41 ± 0.06 (`delta_v`), against 0.09 / 0.00 / 0.14 for PCA(10). Ratio of mean R² over the three features: **7.77** (seed-level CI95 [7.26, 8.06]). EW and peak intensity are linearly decodable by every representation. |
| **LineTransfer** (Hβ → Hα, 2 525 pairs, 1 090 stars) | The equivalent width transfers (R² = 0.66–0.71); line-shape parameters do not. |
| **EWForecast** (MAE ratio to persistence, bootstrap CI95) | PCA(10)+Ridge is the best in-domain baseline: **0.942** [0.927, 0.957]. Chronos-Bolt (small, base) and TimesFM-2.0 also beat persistence zero-shot (best: 0.968); the six Moirai, Moirai-MoE and Time-MoE checkpoints, the BeMAE embedding (1.003) and a causal Transformer (1.17 ± 0.05) do not. A trained linear head on frozen Chronos-Bolt-base reaches 0.932 on the same pairs. |

All numbers, with their provenance, are in `reference_results/` and
`paper/tables/summary.json`.

## Quick reproduction

Two complementary paths are documented in [REPRODUCE.md](REPRODUCE.md):

**Path A — Quick verification.** A single script reproduces the results
that depend on the reference encoder (seed 42) from the public
HuggingFace artefacts and compares them with `reference_results/`:

```bash
pip install -r requirements.txt
bash scripts/reproduce_paper_seed42.sh              # seed 42
ALL_SEEDS=1 bash scripts/reproduce_paper_seed42.sh  # three seeds and the headline ratio
```

Wall-clock: ~25 min on one GPU (RTX 2080 Ti) for seed 42. Outputs land
under `results_reproduced/`.

**Path B — Full pipeline (cluster).** Encoder pre-training for 3 seeds,
ablation grid, multi-seed aggregation and bootstrap CIs:

```bash
cd scripts/cluster && bash run_all.sh
```

Total compute for every result of the paper: ≈20 GPU-hours and
≈1.5 CPU-hours.

## Citation

```bibtex
@inproceedings{bess_bench_2026,
  title     = {BESS-Bench: Benchmarking Spectral Representations for Be-Star Variability},
  author    = {Le Lain, Matthieu and Lefèvre, Sébastien},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

Please also cite the BeSS database: Neiner et al. 2011, AJ 142, 149.

## Licence

CC-BY-4.0 (see `LICENSE`), with attribution to the BeSS database
(<http://basebe.obspm.fr>).
