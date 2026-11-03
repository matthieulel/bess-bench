"""
export_hf.py — export the BeMAE-Hα encoder of a training checkpoint.

A ``best.pt`` written by ``train.py`` holds the encoder and the MAE decoder
(the decoder is only used during pre-training). This script writes the
encoder alone, in the layout of the HuggingFace model repository:

    <output_dir>/
    ├── pytorch_model.bin    # encoder state dict
    ├── config.json          # architecture, input format, training summary
    ├── model.py             # copy of the model definition
    └── config.py            # copy of the configuration classes

The model card (``README.md``) and ``example_usage.py`` of the published
repository are maintained by hand in ``hf_export/bemae-halpha-v1/``.

Usage:
    python export_hf.py --checkpoint runs/Halpha_all_seed42/best.pt \\
        --output_dir hf_export/bemae-halpha-v1
    python export_hf.py --checkpoint runs/Halpha_all_seed123/best.pt \\
        --output_dir hf_export/bemae-halpha-v1/seed123 --weights_only
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from config import HALPHA_CENTER, HALPHA_HALF_WINDOW, N_BINS, ModelConfig
from model import Stage1HalphaModel

TRAINING_KEYS = ("dataset_name", "subset", "batch_size", "epochs", "lr",
                 "weight_decay", "warmup_epochs", "mask_ratio", "seed")


def main():
    parser = argparse.ArgumentParser(description="Export the BeMAE-Hα encoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best.pt written by train.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="matthieulel/bemae-halpha-v1",
                        help="HuggingFace identifier recorded in config.json")
    parser.add_argument("--weights_only", action="store_true",
                        help="Write pytorch_model.bin and config.json only "
                             "(used for the seed123/ and seed456/ sub-folders)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    train_cfg = config.get("train", {})
    model_cfg = ModelConfig(**{k: v for k, v in config.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__})

    full_model = Stage1HalphaModel(model_cfg)
    full_model.load_state_dict(ckpt["model_state_dict"])
    encoder = full_model.encoder
    n_params = sum(p.numel() for p in encoder.parameters())

    torch.save(encoder.state_dict(), out_dir / "pytorch_model.bin")

    meta = {
        "model_type": "bemae-halpha",
        "version": "1.0",
        "model_id": args.model_id,
        "source_checkpoint": f"runs/{ckpt_path.parent.name}/{ckpt_path.name}",
        "seed": train_cfg.get("seed"),
        "n_params_encoder": n_params,
        "input": {
            "n_bins": N_BINS,
            "halpha_center_angstrom": HALPHA_CENTER,
            "halpha_half_window_angstrom": HALPHA_HALF_WINDOW,
            "wavelength_range_angstrom": [HALPHA_CENTER - HALPHA_HALF_WINDOW,
                                          HALPHA_CENTER + HALPHA_HALF_WINDOW],
        },
        "output": {"embedding_dim": model_cfg.d_model, "name": "z_halpha"},
        "model_config": asdict(model_cfg),
        "training": {
            "val_mae_loss": float(ckpt.get("val_mae_loss", 0.0)),
            "epoch_best": int(ckpt.get("epoch", 0)),
            **{k: train_cfg[k] for k in TRAINING_KEYS if k in train_cfg},
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    if not args.weights_only:
        src_dir = Path(__file__).resolve().parent
        shutil.copy(src_dir / "model.py", out_dir / "model.py")
        shutil.copy(src_dir / "config.py", out_dir / "config.py")

    print(f"Encoder exported to {out_dir}  ({n_params:,} parameters, "
          f"seed {train_cfg.get('seed')}, best epoch {meta['training']['epoch_best']})")


if __name__ == "__main__":
    main()
