"""
dataset.py — Temporal dataset for the Stage 2 Hα temporal model.

Builds, for each star, the sequence of (MJD, embedding, EW) read from
star_data.pt and splits it at cfg.cutoff_mjd (train: MJD <= cutoff,
test: MJD > cutoff):
  - stars with fewer than cfg.min_obs observations are discarded
  - sequences longer than cfg.max_seq_len are cut into sliding windows
    of stride cfg.stride
  - training-time augmentation:
    • random subsampling of the sequence (ratio drawn in
      [subsample_ratio_min, subsample_ratio_max])
    • temporal jitter on the MJDs (mjd_jitter_days)
    • random masking of embeddings (mask_ratio)
    • optional Gaussian noise on the embeddings (noise_std)

build_dataloaders() returns the training loader (MJD <= cutoff) and a
validation loader made of the post-cutoff observations of the test stars.

This file is not run on its own: it is imported by train.py and
evaluate.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from config import TemporalTrainConfig


class TemporalDataset(Dataset):
    """
    PyTorch dataset for time series of Hα embeddings.
    Training-time augmentation: subsampling, MJD jitter, masking, noise.
    """

    def __init__(self, star_data, star_names, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.is_train = (split == "train")
        self.windows = []
        self.star_info = []

        for star in star_names:
            data = star_data[star]
            mjds = data["mjds"].numpy()
            embeddings = data["embeddings"].numpy()
            ews = data["ews"].numpy()

            if split == "train":
                mask = mjds <= cfg.cutoff_mjd
            else:
                mask = mjds > cfg.cutoff_mjd

            indices = np.where(mask)[0]
            if len(indices) < 3:
                continue

            sub_mjds = mjds[indices]
            sub_emb = embeddings[indices]
            sub_ews = ews[indices]

            n = len(indices)
            if n <= cfg.max_seq_len:
                self.windows.append({
                    "mjds": sub_mjds,
                    "embeddings": sub_emb,
                    "ews": sub_ews,
                })
                self.star_info.append(star)
            else:
                # Sliding windows of stride cfg.stride
                for start in range(0, n - cfg.max_seq_len + 1, cfg.stride):
                    end = start + cfg.max_seq_len
                    self.windows.append({
                        "mjds": sub_mjds[start:end],
                        "embeddings": sub_emb[start:end],
                        "ews": sub_ews[start:end],
                    })
                    self.star_info.append(star)
                # Last window if the end of the sequence is not covered
                if end < n:
                    self.windows.append({
                        "mjds": sub_mjds[-cfg.max_seq_len:],
                        "embeddings": sub_emb[-cfg.max_seq_len:],
                        "ews": sub_ews[-cfg.max_seq_len:],
                    })
                    self.star_info.append(star)

        print(f"  [{split}] {len(self.windows)} windows, "
              f"{len(set(self.star_info))} stars")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        mjds = window["mjds"].copy()
        emb = window["embeddings"].copy()
        ews = window["ews"].copy()
        n = len(mjds)

        if self.is_train:
            # ── Augmentation 1: random subsampling ──
            if n > 5:
                ratio = np.random.uniform(
                    self.cfg.subsample_ratio_min, self.cfg.subsample_ratio_max
                )
                n_keep = max(4, int(n * ratio))
                if n_keep < n:
                    keep_idx = np.sort(np.random.choice(n, n_keep, replace=False))
                    mjds = mjds[keep_idx]
                    emb = emb[keep_idx]
                    ews = ews[keep_idx]
                    n = n_keep

            # ── Augmentation 2: temporal jitter ──
            if self.cfg.mjd_jitter_days > 0:
                jitter = np.random.uniform(
                    -self.cfg.mjd_jitter_days, self.cfg.mjd_jitter_days, size=n
                )
                mjds = mjds + jitter
                # Keep the MJDs in increasing order
                sort_idx = np.argsort(mjds)
                mjds = mjds[sort_idx]
                emb = emb[sort_idx]
                ews = ews[sort_idx]

            # ── Augmentation 3: embedding masking ──
            # Some embeddings are replaced by zeros (the model has to learn
            # to interpolate). The first and the last point are always kept
            # so that the context is not broken.
            if self.cfg.mask_ratio > 0 and n > 4:
                n_mask = max(1, int(n * self.cfg.mask_ratio))
                # Do not mask the first point nor the last 2
                maskable = np.arange(1, n - 2)
                if len(maskable) > n_mask:
                    mask_idx = np.random.choice(maskable, n_mask, replace=False)
                    emb[mask_idx] = 0.0  # masked embedding

            # ── Augmentation 4: Gaussian noise on the embeddings ──
            noise_std = getattr(self.cfg, 'noise_std', 0.0)
            if noise_std > 0:
                emb = emb + np.random.randn(*emb.shape).astype(np.float32) * noise_std

        # Padding
        T = self.cfg.max_seq_len
        d = emb.shape[1]
        emb_padded = np.zeros((T, d), dtype=np.float32)
        mjd_padded = np.zeros(T, dtype=np.float32)
        ew_padded = np.zeros(T, dtype=np.float32)
        mask = np.zeros(T, dtype=np.float32)

        L = min(n, T)
        emb_padded[:L] = emb[:L]
        mjd_padded[:L] = mjds[:L]
        ew_padded[:L] = ews[:L]
        mask[:L] = 1.0

        return {
            "embeddings": torch.from_numpy(emb_padded),
            "mjds": torch.from_numpy(mjd_padded),
            "ews": torch.from_numpy(ew_padded),
            "mask": torch.from_numpy(mask),
            "length": L,
            "star_name": self.star_info[idx],
        }


def load_star_data(cfg):
    """Load star_data.pt and split the stars into train/test."""
    data_path = Path(cfg.embeddings_dir) / "star_data.pt"
    print(f"Loading embeddings from {data_path}...")
    star_data = torch.load(data_path, weights_only=False)
    print(f"  {len(star_data)} stars loaded")

    train_stars = []
    test_stars = []
    stats = {
        "total_stars": len(star_data),
        "filtered_too_few": 0,
        "filtered_no_test": 0,
    }

    for star, data in star_data.items():
        mjds = data["mjds"].numpy()
        n_total = len(mjds)

        if n_total < cfg.min_obs:
            stats["filtered_too_few"] += 1
            continue

        n_train = (mjds <= cfg.cutoff_mjd).sum()
        n_test = (mjds > cfg.cutoff_mjd).sum()

        if n_train < 3:
            stats["filtered_too_few"] += 1
            continue

        train_stars.append(star)

        if n_test >= cfg.min_test_obs:
            test_stars.append(star)
        else:
            stats["filtered_no_test"] += 1

    stats["train_stars"] = len(train_stars)
    stats["test_stars"] = len(test_stars)

    print(f"  Train : {len(train_stars)} stars")
    print(f"  Test  : {len(test_stars)} stars")
    print(f"  Filtered (too few obs) : {stats['filtered_too_few']}")
    print(f"  Filtered (no test) : {stats['filtered_no_test']}")

    return star_data, {"train_stars": train_stars, "test_stars": test_stars,
                       "stats": stats}


def build_dataloaders(cfg):
    """Build the train and val DataLoaders."""
    star_data, split_info = load_star_data(cfg)

    train_ds = TemporalDataset(star_data, split_info["train_stars"], cfg, split="train")
    val_ds = TemporalDataset(star_data, split_info["test_stars"], cfg, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, split_info
