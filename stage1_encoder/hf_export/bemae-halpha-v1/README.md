---
library_name: pytorch
license: cc-by-4.0
tags:
- astronomy
- spectroscopy
- be-stars
- self-supervised
- masked-autoencoder
- time-series
datasets:
- matthieulel/bess-bench
---

# BeMAE-Hα — masked auto-encoder encoder for Be-star Hα spectra

`matthieulel/bemae-halpha-v1` is the **pre-trained reference encoder** of
*"BESS-Bench: Benchmarking Spectral Representations for Be-Star Variability"*
(NeurIPS 2026, Evaluations & Datasets Track).

It is a small Transformer encoder (≈795 k parameters) that maps a 128-bin
spectrum cropped around the Hα line (6562.8 ± 50 Å) to a 128-dimensional
embedding `z_halpha`. It was trained with a masked auto-encoding objective
(mean squared error on masked patches) on the Hα slice of BESS-Bench
(26,937 spectra). BeMAE is a *reference* encoder for the benchmark, not a
methodological contribution in itself.

## Files

| Path | Content |
|---|---|
| `pytorch_model.bin`, `config.json` | **seed 42**, the reference checkpoint (the one evaluated during peer review, unchanged) |
| `seed123/pytorch_model.bin`, `seed123/config.json` | seed 123 |
| `seed456/pytorch_model.bin`, `seed456/config.json` | seed 456 |
| `model.py`, `config.py` | model definition (identical to `stage1_encoder/` in the code repository) |
| `example_usage.py` | minimal inference script |

The three-seed results of the paper (mean ± std) are computed from these
three checkpoints.

## Quick start

```python
from huggingface_hub import snapshot_download
import sys, json, torch

ckpt = snapshot_download("matthieulel/bemae-halpha-v1")
sys.path.insert(0, ckpt)
from model import SpectralEncoderHalpha, ModelConfig  # noqa: E402

seed_dir = ckpt            # or f"{ckpt}/seed123", f"{ckpt}/seed456"
with open(f"{seed_dir}/config.json") as f:
    cfg = ModelConfig(**json.load(f)["model_config"])

encoder = SpectralEncoderHalpha(cfg)
encoder.load_state_dict(torch.load(f"{seed_dir}/pytorch_model.bin", map_location="cpu"))
encoder.eval()

# flux, wavelengths, validity : torch.Tensor of shape [B, 128]
# Spectra must be cropped to Hα ± 50 Å (6512.8–6612.8 Å, wavelengths in Å)
# and normalised to the pseudo-continuum at the window edges
# (see `example_usage.py` and `stage1_encoder/dataset.py`).
with torch.no_grad():
    z_halpha, *_ = encoder(flux, wavelengths, validity, mask=None)
# z_halpha : Tensor of shape [B, 128]
```

## Training details

| Item | Value |
|---|---|
| Architecture | Transformer encoder, 4 layers, d_model = 128, 4 heads, FFN 512 |
| Input | 128 bins, Hα ± 50 Å (6512.8–6612.8 Å); patches of 8 px, stride 4 → 31 patches |
| Parameters | ≈795 k (encoder; ≈912 k with the decoder used at training time only) |
| Objective | masked auto-encoding, mask ratio 0.60 in 3 contiguous blocks, MSE on masked patches |
| Optimiser | AdamW, lr = 1e-4 (cosine, 5 warm-up epochs), weight decay 0.05, batch 256 |
| Schedule | up to 80 epochs, early stopping (patience 15) at epochs 61 / 76 / 65 |
| Training data | Hα slice of `matthieulel/bess-bench`: 21,781 spectra of 1,455 stars |
| Validation data | 5,156 spectra of 8 held-out stars: GAM CAS, PLEIONE, ZET TAU, V442 AND, PI AQR, DEL SCO, 28 CYG, OME CMA |
| Seeds | 42 (reference), 123, 456 |
| Best validation loss | 0.3354 (seed 42), 0.3351 (seed 123), 0.3348 (seed 456) |
| Hardware | one NVIDIA RTX 2080 Ti per run, 41 / 51 / 43 min |

Pre-training uses no label. Note that the encoder is trained on (unlabelled)
spectra of all stars except the eight held-out ones: the released 70/15/15
star-level partition of the dataset is **not** the encoder's training split.

## Evaluation (BESS-Bench v1.0)

**SpecProbe** — ridge probe on the frozen embedding, 5-fold GroupKFold on
`star_id`, mean ± std over the three seeds (R²):

| Feature | BeMAE `z` (128-D) | PCA(10) | raw flux (128-D) |
|---|---|---|---|
| `fwhm` | **0.508 ± 0.037** | 0.094 | 0.121 |
| `central_depth` | **0.879 ± 0.023** | −0.001 | −0.016 |
| `delta_v` | **0.412 ± 0.056** | 0.139 | 0.188 |
| `vr_ratio` | 0.129 ± 0.068 | 0.117 | 0.063 |

Aggregated ratio of mean R² over the three shape features, `z` vs PCA(10):
**7.77** (bootstrap CI95 over seeds [7.26, 8.06]).

**EWForecast** — the embedding does **not** beat persistence on one-step
EW(Hα) forecasting: `z` + ridge reaches an MAE ratio of 1.003
(CI95 [0.982, 1.026]) where PCA(10) + ridge reaches 0.942.

**LineTransfer** — not evaluated: BeMAE is an Hα-only encoder.

**Latent-space diagnostics** (three seeds): within-star emission/absorption
k-NN accuracy 0.999 ± 0.001 (22 stars observed in both states); star
retrieval Recall@10 = 0.955 ± 0.007 on the eight held-out stars and
0.42 ± 0.01 over all 1,463 stars of the slice.

## Reproducibility

```bash
git clone https://github.com/matthieulel/bess-bench.git
cd bess-bench
bash scripts/reproduce_paper_seed42.sh          # seed-42 results from this checkpoint (~25 min, 1 GPU)
```

Retraining the three seeds from scratch: `scripts/cluster/00a_train_seed.slurm`
(or `python stage1_encoder/train.py --seed <S>`); see `REPRODUCE.md`.

## Limitations

- Trained on BeSS spectra (≈82 % of the released rows come from amateur
  instruments); transfer to other instruments or surveys is not evaluated.
- The input window is fixed at Hα ± 50 Å; other lines require retraining.
- Generalisation to unseen stars is only measured on eight held-out stars.
- `z_halpha` is not a physical parameter vector; a probe is required for
  interpretable regression (see SpecProbe in the paper).

## Citation

```bibtex
@inproceedings{bess_bench_2026,
  title     = {BESS-Bench: Benchmarking Spectral Representations for Be-Star Variability},
  author    = {Le Lain, Matthieu and Lefèvre, Sébastien},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

Please also cite the BeSS database (Neiner et al. 2011, AJ 142, 149).

## Licence

CC-BY-4.0.
