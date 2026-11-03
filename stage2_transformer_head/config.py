"""
config.py — Configuration of the Stage 2 Hα temporal model (causal
Transformer head on BeMAE embeddings).

Two dataclasses:
  - TemporalModelConfig : architecture of the causal Transformer
  - TemporalTrainConfig : data selection, temporal split, optimisation,
    loss weights, augmentation, checkpoints and logging

Main settings:
  - min_obs = 5 : stars with fewer than 5 observations are discarded
  - stride = 32 : stride of the sliding windows used for long sequences
  - Training augmentation: subsampling, embedding masking, MJD jitter
  - Optional loss on the direct prediction of EW_{t+1} (alpha_direct_ew)

This file is not run on its own: it is imported by dataset.py, model.py,
train.py and evaluate.py. Several fields can be overridden from the command
line of train.py (see its argparse options).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalModelConfig:
    """Architecture of the Temporal Aggregator for Hα embeddings."""

    # ── Dimensions ──
    d_input: int = 128
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.15
    max_seq_len: int = 512

    # ── Temporal Positional Encoding ──
    t_ref: float = 48000.0
    t_max: float = 36500.0

    # ── Prediction mode ──
    predict_delta: bool = False   # If True, predict Δz instead of z(t+1)
    condition_dt: bool = False    # If True, condition the next-embedding head on the target ΔT


@dataclass
class TemporalTrainConfig:
    """Training configuration of the Stage 2 Hα temporal model."""

    # ── Data ──
    embeddings_dir: str = "../reference_results/embeddings"
    min_obs: int = 5          # minimum number of observations per star
    max_seq_len: int = 512
    stride: int = 32          # stride of the sliding windows (long sequences)

    # ── Temporal split ──
    cutoff_mjd: float = 59215.0  # 31 Dec 2020
    min_test_obs: int = 3        # min. post-cutoff observations for a test star

    # ── Optimisation ──
    epochs: int = 150         # maximum number of epochs (early stopping in train.py)
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 8
    grad_clip: float = 1.0

    # ── Loss weights ──
    alpha_next: float = 1.0       # MSE next-embedding
    alpha_contr: float = 0.0      # InfoNCE contrastive
    alpha_ew: float = 0.0         # MSE on EW_{t+1} from the auxiliary EW head
    alpha_direct_ew: float = 0.0  # direct prediction of EW_{t+1}
    contrastive_tau: float = 0.07

    # ── Augmentation ──
    subsample_ratio_min: float = 0.6   # lower bound of the random subsampling ratio
    subsample_ratio_max: float = 1.0
    mask_ratio: float = 0.15           # mask 15% of the embeddings
    mjd_jitter_days: float = 0.5       # ±0.5 day of temporal jitter
    noise_std: float = 0.0             # Gaussian noise on the embeddings (0 = disabled)

    # ── Checkpoints ──
    save_every: int = 10
    output_dir: str = "runs"
    run_name: Optional[str] = None

    # ── wandb logging ──
    wandb_project: str = "bess-bench"
    wandb_entity: Optional[str] = None

    # ── Misc ──
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    # ── Path to the Stage 1 Hα checkpoint ──
    stage1_checkpoint: str = "../stage1_encoder/runs/Halpha_all_seed42/best.pt"
