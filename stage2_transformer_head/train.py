"""
train.py — Training loop of the Stage 2 Hα temporal model (causal
Transformer head on BeMAE embeddings).

  - Total loss = α_next · next-embedding MSE + α_contr · InfoNCE
    + α_ew · auxiliary EW loss + α_direct_ew · direct EW_{t+1} loss
  - val/direct_ew_mae is logged for monitoring
  - Model selection and early stopping use the total loss on the
    validation loader, i.e. on the post-cutoff observations of the test
    stars (patience: 30 epochs)
  - Writes config.json, best.pt and periodic checkpoints to
    runs/<run_name>/ and logs the metrics to Weights & Biases (wandb)

Usage:
    cd stage2_transformer_head
    # configuration of the paper (see README.md)
    python train.py --run_name tfm_seed42 --seed 42 --n_layers 4 --dropout 0.1 --predict_delta --condition_dt
    # other variants
    python train.py --run_name base
    python train.py --run_name direct_ew --alpha_direct_ew 2.0

--embeddings_dir is the directory that contains star_data.pt (default:
../reference_results/embeddings, the seed-42 BeMAE embeddings).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from config import TemporalModelConfig, TemporalTrainConfig
from dataset import build_dataloaders
from model import (TemporalAggregator, next_embedding_loss,
                    contrastive_loss, ew_prediction_loss, direct_ew_loss,
                    EW_SCALE)


# ══════════════════════════════════════════════════════════════════════════════
# COSINE SCHEDULER WITH WARMUP
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(optimizer, epoch: int, cfg: TemporalTrainConfig):
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
    else:
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def persistence_mse(embeddings: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask[:, :-1] * mask[:, 1:]
    pred = embeddings[:, :-1]
    target = embeddings[:, 1:]
    mse = ((pred - target) ** 2).mean(dim=-1)
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0
    return float((mse * valid).sum() / n_valid)


def mean_cosine_sim(z_pred: torch.Tensor, z_target: torch.Tensor,
                     mask: torch.Tensor) -> float:
    valid = mask[:, :-1] * mask[:, 1:]
    pred = z_pred[:, :-1]
    target = z_target[:, 1:]
    cos = F.cosine_similarity(pred, target, dim=-1)
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0
    return float((cos * valid).sum() / n_valid)


def direct_ew_mae(ew_direct: torch.Tensor, ews: torch.Tensor,
                   mask: torch.Tensor) -> float:
    """MAE of the direct EW prediction, for monitoring."""
    valid = mask[:, :-1] * mask[:, 1:]
    pred = ew_direct[:, :-1]
    target = ews[:, 1:]
    not_nan = ~torch.isnan(target)
    ew_valid = (valid > 0) & not_nan
    n_valid = ew_valid.sum()
    if n_valid == 0:
        return 0.0
    return float(torch.abs(pred[ew_valid] - target[ew_valid]).mean())


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL EPOCH
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    totals = {k: 0.0 for k in [
        "loss", "loss_next", "loss_contr", "loss_ew", "loss_direct_ew",
        "cos_sim", "persist_mse", "grad_norm", "direct_ew_mae",
    ]}
    n_batches = 0

    for batch in loader:
        emb = batch["embeddings"].to(device)
        mjds = batch["mjds"].to(device)
        mask = batch["mask"].to(device)
        ews = batch["ews"].to(device)

        out = model(emb, mjds, mask)

        l_next = next_embedding_loss(out["z_pred"], emb, mask)
        l_contr = contrastive_loss(out["proj"], mask, cfg.contrastive_tau)
        l_ew = ew_prediction_loss(out["ew_pred"], ews, mask)
        l_direct_ew = direct_ew_loss(out["ew_direct"], ews, mask)

        loss = (cfg.alpha_next * l_next
                + cfg.alpha_contr * l_contr
                + cfg.alpha_ew * l_ew
                + cfg.alpha_direct_ew * l_direct_ew)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["loss_next"] += l_next.item()
        totals["loss_contr"] += l_contr.item()
        totals["loss_ew"] += l_ew.item()
        totals["loss_direct_ew"] += l_direct_ew.item()
        totals["cos_sim"] += mean_cosine_sim(out["z_pred"].detach(), emb, mask)
        totals["persist_mse"] += persistence_mse(emb, mask)
        totals["grad_norm"] += float(grad_norm)
        totals["direct_ew_mae"] += direct_ew_mae(
            out["ew_direct"].detach(), ews, mask
        )
        n_batches += 1

    n = max(n_batches, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    totals = {k: 0.0 for k in [
        "loss", "loss_next", "loss_contr", "loss_ew", "loss_direct_ew",
        "cos_sim", "persist_mse", "direct_ew_mae",
    ]}
    n_batches = 0

    for batch in loader:
        emb = batch["embeddings"].to(device)
        mjds = batch["mjds"].to(device)
        mask = batch["mask"].to(device)
        ews = batch["ews"].to(device)

        out = model(emb, mjds, mask)

        l_next = next_embedding_loss(out["z_pred"], emb, mask)
        l_contr = contrastive_loss(out["proj"], mask, cfg.contrastive_tau)
        l_ew = ew_prediction_loss(out["ew_pred"], ews, mask)
        l_direct_ew = direct_ew_loss(out["ew_direct"], ews, mask)

        loss = (cfg.alpha_next * l_next
                + cfg.alpha_contr * l_contr
                + cfg.alpha_ew * l_ew
                + cfg.alpha_direct_ew * l_direct_ew)

        totals["loss"] += loss.item()
        totals["loss_next"] += l_next.item()
        totals["loss_contr"] += l_contr.item()
        totals["loss_ew"] += l_ew.item()
        totals["loss_direct_ew"] += l_direct_ew.item()
        totals["cos_sim"] += mean_cosine_sim(out["z_pred"], emb, mask)
        totals["persist_mse"] += persistence_mse(emb, mask)
        totals["direct_ew_mae"] += direct_ew_mae(out["ew_direct"], ews, mask)
        n_batches += 1

    n = max(n_batches, 1)
    metrics = {k: v / n for k, v in totals.items()}

    if metrics["persist_mse"] > 0:
        metrics["next_vs_persist"] = metrics["loss_next"] / metrics["persist_mse"]
    else:
        metrics["next_vs_persist"] = float("inf")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(model_cfg: TemporalModelConfig, train_cfg: TemporalTrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    train_loader, val_loader, split_info = build_dataloaders(train_cfg)

    model = TemporalAggregator(model_cfg).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    run_dir = Path(train_cfg.output_dir) / (train_cfg.run_name or "default")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": vars(model_cfg),
        "train": vars(train_cfg),
        "n_params": n_params,
        "split_info": split_info,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    wandb.init(
        project=train_cfg.wandb_project,
        entity=train_cfg.wandb_entity,
        name=train_cfg.run_name or "stage2_temporal",
        config=config_dict,
        dir=str(run_dir),
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 30  # early-stopping patience (epochs without val-loss improvement)
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Stage 2 Hα — {train_cfg.epochs} epochs")
    print(f"  min_obs={train_cfg.min_obs}, stride={train_cfg.stride}, "
          f"mask_ratio={train_cfg.mask_ratio}, mjd_jitter={train_cfg.mjd_jitter_days}d")
    print(f"  α_next={train_cfg.alpha_next}, α_contr={train_cfg.alpha_contr}, "
          f"α_ew={train_cfg.alpha_ew}, α_direct_ew={train_cfg.alpha_direct_ew}")
    print(f"  Device : {device}")
    print(f"{'='*70}\n")

    for epoch in range(train_cfg.epochs):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, train_cfg)

        train_m = train_epoch(model, train_loader, optimizer, train_cfg, device)
        val_m = val_epoch(model, val_loader, train_cfg, device)

        dt = time.time() - t0

        log_dict = {
            "epoch": epoch, "lr": lr,
            "train/loss": train_m["loss"],
            "train/loss_next": train_m["loss_next"],
            "train/loss_contr": train_m["loss_contr"],
            "train/loss_ew": train_m["loss_ew"],
            "train/loss_direct_ew": train_m["loss_direct_ew"],
            "train/cos_sim": train_m["cos_sim"],
            "train/persist_mse": train_m["persist_mse"],
            "train/grad_norm": train_m["grad_norm"],
            "train/direct_ew_mae": train_m["direct_ew_mae"],
            "val/loss": val_m["loss"],
            "val/loss_next": val_m["loss_next"],
            "val/loss_contr": val_m["loss_contr"],
            "val/loss_ew": val_m["loss_ew"],
            "val/loss_direct_ew": val_m["loss_direct_ew"],
            "val/cos_sim": val_m["cos_sim"],
            "val/persist_mse": val_m["persist_mse"],
            "val/next_vs_persist": val_m["next_vs_persist"],
            "val/direct_ew_mae": val_m["direct_ew_mae"],
            "time/epoch_sec": dt,
        }
        wandb.log(log_dict, step=epoch)

        # Selection criterion: total val_loss (not only the next-embedding term)
        improve = "★" if val_m["loss"] < best_val_loss else " "
        print(
            f"[{epoch:3d}/{train_cfg.epochs}] "
            f"train: L={train_m['loss']:.4f} "
            f"(next={train_m['loss_next']:.4f} dew={train_m['loss_direct_ew']:.4f}) "
            f"| val: L={val_m['loss']:.4f} "
            f"(next={val_m['loss_next']:.4f} dew={val_m['loss_direct_ew']:.4f} "
            f"ew_mae={val_m['direct_ew_mae']:.2f}) "
            f"cos={val_m['cos_sim']:.3f} "
            f"| lr={lr:.2e} {improve} ({dt:.1f}s)"
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_m,
                "train_metrics": train_m,
            }, run_dir / "best.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % train_cfg.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, run_dir / f"checkpoint_{epoch}.pt")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"\n{'='*70}")
    print(f"Training finished — best model: epoch {best_epoch}")
    print(f"  Saved to {run_dir / 'best.pt'}")
    print(f"{'='*70}")

    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 Hα — Temporal Aggregator Training"
    )

    # Model config
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--predict_delta", action="store_true",
                        help="Predict Δz instead of z(t+1) — residual skip connection")
    parser.add_argument("--condition_dt", action="store_true",
                        help="Condition the prediction on the target ΔT")
    # Training config
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha_next", type=float, default=None)
    parser.add_argument("--alpha_contr", type=float, default=None)
    parser.add_argument("--alpha_ew", type=float, default=None)
    parser.add_argument("--alpha_direct_ew", type=float, default=None)
    parser.add_argument("--contrastive_tau", type=float, default=None)
    parser.add_argument("--min_obs", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--mjd_jitter_days", type=float, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--subsample_ratio_min", type=float, default=None)
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42 from config)")

    args = parser.parse_args()

    model_cfg = TemporalModelConfig()
    train_cfg = TemporalTrainConfig()

    # Model
    for field_name in ["d_model", "n_layers", "n_heads", "d_ff", "dropout"]:
        val = getattr(args, field_name)
        if val is not None:
            setattr(model_cfg, field_name, val)

    # predict_delta and condition_dt are booleans (store_true)
    if args.predict_delta:
        model_cfg.predict_delta = True
    if args.condition_dt:
        model_cfg.condition_dt = True

    for field_name in ["run_name", "epochs", "batch_size", "lr",
                        "alpha_next", "alpha_contr", "alpha_ew", "alpha_direct_ew",
                        "contrastive_tau", "min_obs", "stride",
                        "mask_ratio", "mjd_jitter_days", "noise_std",
                        "subsample_ratio_min",
                        "embeddings_dir", "wandb_project", "seed"]:
        val = getattr(args, field_name)
        if val is not None:
            setattr(train_cfg, field_name, val)

    train(model_cfg, train_cfg)
