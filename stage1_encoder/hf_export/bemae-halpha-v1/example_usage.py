"""
example_usage.py — Minimal inference with BeMAE-Hα.

This script downloads the checkpoint from the HuggingFace Hub and computes
the z_halpha embedding of a dummy batch with the expected input format
(128 bins over Hα ± 50 Å, flux normalised to the pseudo-continuum).
See `stage1_encoder/dataset.py` in the code repository for the
preprocessing of real BESS-Bench spectra.
"""
import json
import sys

import torch
from huggingface_hub import snapshot_download

MODEL_ID = "matthieulel/bemae-halpha-v1"

# 1. Download the snapshot
ckpt_dir = snapshot_download(MODEL_ID)
sys.path.insert(0, ckpt_dir)

from model import SpectralEncoderHalpha, ModelConfig  # noqa: E402

# 2. Load config and weights (seed 42 at the root; seed123/ and seed456/ also available)
seed_dir = ckpt_dir
with open(f"{seed_dir}/config.json") as f:
    meta = json.load(f)
cfg = ModelConfig(**meta["model_config"])

encoder = SpectralEncoderHalpha(cfg)
state = torch.load(f"{seed_dir}/pytorch_model.bin", map_location="cpu")
encoder.load_state_dict(state)
encoder.eval()

# 3. Prepare a dummy spectrum (Hα-centred, 128 bins, pseudo-continuum at 1.0)
B = 4
flux = torch.ones(B, 128) + 0.3 * torch.randn(B, 128) * 0.01
wavelengths = torch.linspace(6512.8, 6612.8, 128).unsqueeze(0).expand(B, -1)
validity = torch.ones(B, 128)

# 4. Inference
with torch.no_grad():
    z_halpha, *_ = encoder(flux, wavelengths, validity, mask=None)

print(f"z_halpha shape : {tuple(z_halpha.shape)}")
print(f"z_halpha[0, :8] : {z_halpha[0, :8].numpy()}")
