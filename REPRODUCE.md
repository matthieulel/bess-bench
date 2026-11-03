# Reproducing BESS-Bench results

This document describes how to reproduce every number reported in the paper,
**from a fresh clone**, with two complementary paths:

| Path | Hardware | Time | Coverage |
|---|---|---|---|
| **A. Quick verification** | 1 GPU (RTX 2080 Ti or equivalent), or CPU fallback | ~25 min on GPU / ~3 h on CPU | seed 42 from the public checkpoint; `ALL_SEEDS=1` adds seeds 123 / 456 and the three-seed headline ratio |
| **B. Full pipeline** | 1 GPU + 8 CPU on a SLURM cluster | ≈20 GPU-h + ≈1.5 CPU-h in total | encoder pre-training (3 seeds), ablations, every task, bootstrap CIs, tables and figures |

The two paths share the same code base; path A starts from the released
checkpoints and skips encoder pre-training and the ablation grid.

> **Note on the SLURM scripts.** The settings shared by the files of
> `scripts/cluster/` are in `scripts/cluster/_common.sh` (conda
> environment through `BESS_CONDA_ENV`, Weights & Biases off by default).
> Adapt the `#SBATCH --partition` lines to your cluster; the Python
> dependencies are listed in `requirements.txt`.

---

## Path A -- Quick verification (single command)

### 1. Clone and create environment

```bash
git clone https://github.com/matthieulel/bess-bench.git && cd bess-bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the needed packages (`torch`, `datasets`,
`huggingface_hub`, `scikit-learn`, `numpy`, `scipy`, `astropy`,
`matplotlib`, `chronos-forecasting`, `timesfm`, `wandb`). Tested with
Python 3.10 and 3.11.

> **Note.** `chronos-forecasting` and `timesfm` are only required for
> the EWForecast TS-FM baselines; run with `SKIP_TSFM=1` if you only need
> SpecProbe / LineTransfer / PCA+Ridge. The additional forecasters
> (`EXTRA_TSFM=1`) need more packages: `transformers` for Time-MoE and
> `uni2ts` for Moirai / Moirai-MoE. `uni2ts` pins its own versions of
> torch; install it in a separate environment
> (`python -m venv --system-site-packages venv_moirai && pip install uni2ts`)
> and run `downstream/t3_moirai_zeroshot.pyA` and
> `downstream/t3_moiraimoe_zeroshot.py` from there.

### 2. Run the reproduction script

```bash
bash scripts/reproduce_paper_seed42.sh
```

Device selection is automatic: GPU (CUDA, bf16 for TS-FM inference) when
available -- this matches the reference run -- otherwise CPU (deterministic
fp32 fallback). Override explicitly with `DEVICE=cpu bash scripts/reproduce_paper_seed42.sh`
or `DEVICE=cuda ...`. Other options: `SKIP_TSFM=1` (no foundation model),
`ALL_SEEDS=1` (also seeds 123 and 456, and the three-seed headline
ratio), `EXTRA_TSFM=1` (trained head on Chronos-Bolt, Time-MoE, Moirai,
Moirai-MoE when their packages are installed), `MODEL_DIR=<dir>` (local
copy of the model repository).

The script:

1. downloads the encoder checkpoint `matthieulel/bemae-halpha-v1`
   (3 MB per seed) and the BESS-Bench dataset
   `matthieulel/bess-bench` (10 GB, cached locally, first run only);
2. recomputes the encoder embeddings on the selected device;
3. extracts the H-alpha and H-beta classical features;
4. runs SpecProbe, LineTransfer, the EWForecast baselines (PCA+Ridge,
   z+Ridge, time-aware ridge, validation-based configuration selection,
   zero-shot Chronos-Bolt-S/B and TimesFM-2.0) and the per-SNR-bin
   diagnostic;
5. regenerates the figures and tables that depend on these outputs;
6. **calls `compare_with_reference.py`** at the end and prints a
   per-metric report.

All outputs land under `results_reproduced/`:

```
results_reproduced/
├── assets/hf_model/               # downloaded BeMAE checkpoint(s)
├── embeddings/star_data.pt        # recomputed encoder outputs (seed 42; seed123/, seed456/ with ALL_SEEDS=1)
├── features/spectral_features.pt  # Hα features (EW, FWHM, V/R, …)
├── features/spectral_features_hbeta.pt
├── downstream/
│   ├── probe_features_results_seed42.json
│   ├── probe_t2_hbeta_to_halpha.json
│   ├── t3_pca_ridge_temporal_baseline.json
│   ├── t3_z_ridge_temporal_baseline_ctx{5,10}.json
│   ├── t3_time_aware_ridge.json
│   ├── t3_pca_ridge_select_config.json
│   ├── t3_ts_fm_baselines.json
│   ├── snr_ablation_seed42.json
│   └── sigma_ew_and_scree.json
├── stats/t1_multiseed_summary.json  # with ALL_SEEDS=1
├── figures/, tables/              # regenerated figures and LaTeX tables
├── compare_report.txt             # diff vs reference_results/
└── logs/reproduce.log
```

### 3. Read the reproducibility report

The report (also written to `results_reproduced/compare_report.txt`) flags
each metric:

| Flag | Tolerance | Meaning |
|---|---|---|
| `STRICT` | $\|\Delta\| < 10^{-3}$ | reproduction to numerical noise |
| `GPU-OK` | $\|\Delta\| < 10^{-2}$ | acceptable numerical noise (e.g.\ bf16 vs fp32) |
| `MISMATCH` | $\|\Delta\| > 10^{-2}$ | flagged for inspection |

Steps that were not run (optional forecasters, other seeds) are listed as
skipped. Profile obtained for the camera-ready release with
`ALL_SEEDS=1 EXTRA_TSFM=1` on one RTX 2080 Ti, in an environment different
from the one that produced `reference_results/` (scikit-learn 1.8 vs 1.7):

| Block | # metrics | STRICT | largest deviation |
|---|---:|---:|---:|
| SpecProbe, seed 42 | 72 | 72 | 1.1e-4 |
| LineTransfer | 36 | 36 | 1.6e-7 |
| EWForecast, PCA+Ridge | 24 | 24 | 0 |
| EWForecast, BeMAE z + Ridge | 38 | 38 | 0 |
| EWForecast, time-aware ridge | 15 | 15 | 2.2e-11 |
| EWForecast, validation-based configuration selection | 16 | 16 | 5.9e-5 |
| EWForecast, Chronos-Bolt / TimesFM-2.0 | 9 | 9 | 4.9e-5 |
| EWForecast, well-sampled / sparsely sampled stars | 27 | 27 | 6.6e-5 |
| EWForecast, Chronos head and Time-MoE (`EXTRA_TSFM=1`) | 20 | 20 | 1.6e-6 |
| SNR diagnostic, seed 42 | 60 | 60 | 3.1e-6 |
| SpecProbe and SNR, seeds 123 / 456, headline ratio (`ALL_SEEDS=1`) | 267 | 267 | 1.1e-4 |
| **Total** | **584** | **584** | **1.1e-4** |

The Moirai and Moirai-MoE scripts were not part of this run (`uni2ts` lives
in a separate environment). The largest deviations are on the PCA(32) and
PCA(50) baselines, whose randomized SVD differs slightly across library
versions; the three-seed headline ratio is reproduced as 7.767
[7.258, 8.060].

### 4. Determinism settings

The script pins everything that can be pinned:

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=disabled
```

When `DEVICE=cpu`, all linear algebra runs in deterministic fp32. When
`DEVICE=cuda`, only the TS-FM inference path uses bf16 (matching the
reference run); BeMAE encoder remains in fp32.

---

## Path B -- Full pipeline (multi-seed, ablations, GPU)

### 1. SLURM submission (recommended on a cluster)

```bash
cd scripts/cluster
bash run_all.sh
```

This submits the following SLURM jobs, chained by `--dependency=afterok`:

| # | Phase | Resources | Duration | What it does |
|---|------|-----------|----------|--------------|
| 0a | `00a_train_seed.slurm` (× 3) | 1 GPU, 8 CPU, 32 GB | 41–51 min/seed | Pre-trains BeMAE-H-alpha for one seed (42, 123, 456 in parallel). |
| 0 | `00_pretrain_encoder.slurm`  | 1 GPU, 8 CPU, 32 GB | ~20 min | Latent-space diagnostics, embeddings and HuggingFace export for the three seeds. Waits on 0a. |
| 1 | `01_extract_features.slurm`  | 8 CPU, 32 GB        | ~30 min–1 h | Extracts the H-alpha and H-beta spectral features. Runs in parallel with phase 0. |
| 2a | `02a_downstream_cpu.slurm`  | 8 CPU, 32 GB        | ~45 min | SpecProbe and SNR diagnostic × 3 seeds, LineTransfer, EWForecast ridge baselines. Waits on 0 + 1. |
| 2b | `02b_ts_fm_gpu.slurm`       | 1 GPU, 4 CPU, 32 GB | ~30 min–1 h | EWForecast time-series foundation models. Waits on 0 + 1. |
| 3 | `03_aggregate_paper.slurm`   | 4 CPU, 16 GB        | ~10 min | Multi-seed aggregation, bootstrap CI95, LaTeX tables, figures. Waits on all. |

`WITH_ABLATIONS=1 bash run_all.sh` also submits the ablation grid:
`00b_ablations_pretrain.slurm` (24 trainings, ≈16 GPU-h) and
`02c_ablations_probe.slurm`.

Total compute for every result of the paper, ablations included: ≈20 GPU-h + ≈1.5 CPU-h.

**Outputs.** The pipeline writes where the analysis scripts read by default:
it regenerates `reference_results/` and `paper/tables/` **in place**. Run it on
a branch or a copy; `git diff reference_results` then shows how your run
differs from the numbers of the paper. Encoder pre-training on a GPU is not
bit-reproducible, so a retrained seed gives slightly different numbers; the
released checkpoints are the reference for the numbers of the paper.

Logs land in `scripts/cluster/logs/<phase>_<jobid>.{out,err}`.

### 2. Without SLURM (single workstation, 1 GPU)

Every `.slurm` script is a plain Bash script -- the `#SBATCH` lines are just
comments. Run sequentially:

```bash
cd scripts/cluster
bash 00_pretrain_encoder.slurm      # trains the three seeds if their checkpoints are missing
bash 01_extract_features.slurm
bash 02a_downstream_cpu.slurm
bash 02b_ts_fm_gpu.slurm
bash 03_aggregate_paper.slurm
```

> **Why ship the cluster scripts?** They document, at the command level,
> the sequence that produces the numbers of `reference_results/`. Path A
> (`reproduce_paper_seed42.sh`) is a subset of the same pipeline that
> starts from the released checkpoints.

### 3. Re-running a single phase

Every phase can be re-run on its own. Training is skipped when the
checkpoint (`stage1_encoder/runs/<run>/best.pt`) already exists; the
other steps overwrite their outputs.

### 4. Causal Transformer head (EWForecast, negative result)

The Transformer head of the last row of the EWForecast table is in
`stage2_transformer_head/` (see its README); it is trained and scored
separately from the pipeline above (≈0.6 GPU-h for three seeds).

---

## Reference results layout

The directory `reference_results/` ships the canonical paper numbers for
diff-based verification:

```
reference_results/
├── stage1_encoder/
│   ├── multiseed_summary.json                          # latent-space diagnostics, 3 seeds
│   └── Halpha_all_seed{42,123,456}/eval_results.json
├── embeddings/                                         # star_data.pt (seed 42), seed123/, seed456/
├── features/                                           # spectral_features.pt, spectral_features_hbeta.pt
├── downstream/
│   ├── probe_features_results_seed{42,123,456}.json    # SpecProbe
│   ├── probe_features_results_ablation_*.json          # 24 ablation runs
│   ├── snr_ablation_seed{42,123,456}.json              # per-SNR-bin diagnostic
│   ├── probe_t2_hbeta_to_halpha.json                   # LineTransfer
│   ├── t3_pca_ridge_temporal_baseline.json             # EWForecast, PCA+Ridge
│   ├── t3_z_ridge_temporal_baseline_ctx{5,10}.json     # EWForecast, BeMAE z+Ridge
│   ├── t3_ts_fm_baselines.json                         # EWForecast, Chronos-Bolt / TimesFM-2.0
│   ├── t3_{moirai,moiraimoe,timemoe}_zeroshot.json     # EWForecast, other zero-shot models
│   ├── t3_chronos_{trained_head,zeroshot_samepairs}.json
│   ├── t3_time_aware_ridge.json, t3_pca_ridge_select_config.json
│   ├── t3_well_sampled_stratification.json             # EWForecast, stars with ≥ 30 epochs vs others
│   ├── t3_transformer_head.json                        # EWForecast, causal Transformer (3 seeds)
│   ├── sigma_ew_and_scree.json                         # EW noise floor, PCA scree
│   ├── spectral_features_stats.json
│   └── spectral_features_hbeta.manifest.json
├── stats/
│   ├── t1_multiseed_summary.json                       # SpecProbe across seeds, headline ratio
│   ├── ablations_summary.json                          # ablation grid
│   ├── t3_bootstrap_cis.json                           # CI95 EWForecast
│   └── coverage_audit.json                             # spectral coverage
└── submission_ungrouped/                               # outputs of the submitted manuscript
                                                        # (spectrum-level folds, see CHANGELOG.md)
```

All files are produced by phases 0–3 of path B; path A regenerates the
seed-42 subset and `compare_with_reference.py` flags any divergence.

---

## Collection pipeline (not reproduced)

The harvesting code that downloads raw FITS from the BeSS SSA endpoint is
**not** redistributed — it depends on an external VO service and
re-downloading 339 k spectra takes ≈ 30 h. The full collection protocol is
described in the paper (`Section: Data collection`) and in
`dataset/DATASHEET.md`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: chronos` / `timesfm` | `pip install chronos-forecasting timesfm` (only needed for EWForecast TS-FM) |
| `OOM` during encoder pre-training | Pass a smaller `--batch_size` to `stage1_encoder/train.py` (default 256) |
| TimesFM hangs on CPU | Expected — single TimesFM zero-shot run takes ~10 min on CPU. Reduce `--batch_size 16` if you have <8 GB RAM. |
| `compare_with_reference.py` flags deviations of order 1e-4 on PCA(32) / PCA(50) cells | Expected: randomized SVD differs slightly across library versions and thread counts. |
| Wandb prompts for login | The script sets `WANDB_MODE=disabled`; if a script ignores it, run `wandb disabled` once. |
