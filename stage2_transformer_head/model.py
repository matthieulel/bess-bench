"""
model.py — Temporal Aggregator (Stage 2 Hα) with causal attention: a small
causal Transformer over sequences of BeMAE Hα embeddings.

Contents:
  - TemporalPE : sinusoidal positional encoding based on the actual MJD
  - CausalTemporalTransformer : backbone with causal attention
  - Prediction heads : NextEmbeddingHead, ContrastiveHead, EWHead and
    DirectEWHead (direct prediction of EW_{t+1}: the model learns to
    predict the physical EW value of the next spectrum without going
    through an intermediate embedding + post-hoc Ridge)
  - TemporalAggregator : the full model
  - Loss functions; the EW losses are normalised by EW_SCALE

This file is not run on its own: it is imported by train.py and
evaluate.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TemporalModelConfig


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

class TemporalPE(nn.Module):
    """
    Sinusoidal positional encoding based on the actual (irregular) MJD.
    """

    def __init__(self, d_model: int = 128, t_ref: float = 48000.0,
                 t_max: float = 36500.0):
        super().__init__()
        self.d_model = d_model
        self.t_ref = t_ref
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (math.log(t_max) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, mjds: torch.Tensor) -> torch.Tensor:
        dt = (mjds - self.t_ref).unsqueeze(-1)
        pe = torch.zeros(*mjds.shape, self.d_model, device=mjds.device)
        pe[..., 0::2] = torch.sin(dt / self.div_term)
        pe[..., 1::2] = torch.cos(dt / self.div_term)
        return pe


# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL TEMPORAL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class CausalTemporalTransformer(nn.Module):
    """
    Causal Transformer for time series of Hα embeddings.
    """

    def __init__(self, cfg: TemporalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.d_input, cfg.d_model)
        self.temporal_pe = TemporalPE(cfg.d_model, cfg.t_ref, cfg.t_max)
        self.pre_norm = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(T, device=device)

    def forward(self, embeddings: torch.Tensor, mjds: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = embeddings.shape

        # L2 normalisation of the Hα embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        x = self.input_proj(embeddings)
        x = x + self.temporal_pe(mjds)
        x = self.pre_norm(x)

        causal_mask = self._make_causal_mask(T, x.device)
        key_padding_mask = (padding_mask == 0)

        h = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=key_padding_mask
        )
        return self.output_norm(h)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HEADS
# ══════════════════════════════════════════════════════════════════════════════

class NextEmbeddingHead(nn.Module):
    """Prediction of the next embedding z_{t+1} from h_t.

    Options:
      - condition_dt: concatenates the ΔT embedding (32 dims) to the input
    """

    def __init__(self, d_model: int, d_output: int,
                 condition_dt: bool = False):
        super().__init__()
        self.condition_dt = condition_dt
        d_input = d_model + (32 if condition_dt else 0)
        self.head = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_output),
        )

    def forward(self, h: torch.Tensor, dt_emb: torch.Tensor = None):
        if self.condition_dt and dt_emb is not None:
            h = torch.cat([h, dt_emb], dim=-1)
        return self.head(h)


class ContrastiveHead(nn.Module):
    """Projection for InfoNCE."""

    def __init__(self, d_model: int, d_proj: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_proj)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(h), dim=-1)


class EWHead(nn.Module):
    """Auxiliary prediction of EW(Hα) at the next step, from h_t."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)


class DirectEWHead(nn.Module):
    """
    DIRECT prediction of EW_{t+1} from h_t.

    This head learns end-to-end to predict the physical EW value of the
    next spectrum, as opposed to the post-hoc route (predicted embedding
    followed by a Ridge regression z → EW, see evaluate.py).

    Deeper architecture than EWHead because it is meant as a main task,
    not as an auxiliary one.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAggregator(nn.Module):
    """
    Stage 2 Hα — Temporal Aggregator.

    Pipeline:
      embeddings [B, T, 128] + MJDs [B, T]
      → CausalTemporalTransformer → h [B, T, 128]
      → NextEmbeddingHead(h_t) → ẑ_{t+1}     (MSE loss, α_next)
      → ContrastiveHead(h_t) → p_t            (InfoNCE loss, α_contr)
      → EWHead(h_t) → EW_aux                  (MSE loss, α_ew)
      → DirectEWHead(h_t) → EW_direct_{t+1}   (Smooth L1 loss, α_direct_ew)

    Options:
      - predict_delta: skip connection z_pred = z(t) + Δ
      - condition_dt: conditions NextEmbeddingHead on ΔT = mjd(t+1) - mjd(t)

    ~500K parameters with the default configuration.
    """

    def __init__(self, cfg: TemporalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = CausalTemporalTransformer(cfg)
        self.next_head = NextEmbeddingHead(
            cfg.d_model, cfg.d_input,
            condition_dt=getattr(cfg, 'condition_dt', False),
        )
        self.contrastive_head = ContrastiveHead(cfg.d_model, d_proj=64)
        self.ew_head = EWHead(cfg.d_model)
        self.direct_ew_head = DirectEWHead(cfg.d_model)

        # ΔT conditioning: encodes log(ΔT) into a 32-dim vector
        if getattr(cfg, 'condition_dt', False):
            self.dt_encoder = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, 32),
            )
        else:
            self.dt_encoder = None

    def forward(self, embeddings, mjds, mask):
        h = self.backbone(embeddings, mjds, mask)

        # Compute the ΔT embedding if condition_dt is set
        dt_emb = None
        if self.dt_encoder is not None:
            # ΔT = mjd(t+1) - mjd(t) for each position t
            # Shift the mjds: dt[t] = mjd[t+1] - mjd[t]
            dt = torch.zeros_like(mjds)
            dt[:, :-1] = mjds[:, 1:] - mjds[:, :-1]
            dt[:, -1] = dt[:, -2].clone() if mjds.shape[1] > 1 else 1.0
            # Clamp and log-transform for stability (ΔT ranges from 0.1 to 3000+ days)
            dt = torch.clamp(dt, min=0.1)
            dt_log = torch.log(dt).unsqueeze(-1)  # [B, T, 1]
            dt_emb = self.dt_encoder(dt_log)       # [B, T, 32]

        if self.cfg.predict_delta:
            delta = self.next_head(h, dt_emb)
            z_pred = embeddings + delta
        else:
            z_pred = self.next_head(h, dt_emb)

        proj = self.contrastive_head(h)
        ew_pred = self.ew_head(h)
        ew_direct = self.direct_ew_head(h)
        return {
            "h": h,
            "z_pred": z_pred,
            "proj": proj,
            "ew_pred": ew_pred,
            "ew_direct": ew_direct,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def next_embedding_loss(z_pred, z_target, mask):
    valid = mask[:, :-1] * mask[:, 1:]
    pred = z_pred[:, :-1]
    target = z_target[:, 1:]
    mse = ((pred - target) ** 2).mean(dim=-1)
    n_valid = valid.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=z_pred.device)
    return (mse * valid).sum() / n_valid


EW_SCALE = 50.0


def ew_prediction_loss(ew_pred, ews, mask):
    """Auxiliary EW loss (EWHead): MSE on EW_{t+1}, normalised by EW_SCALE."""
    valid = mask[:, :-1] * mask[:, 1:]
    pred = ew_pred[:, :-1]
    target = ews[:, 1:]
    not_nan = ~torch.isnan(target)
    ew_valid = (valid > 0) & not_nan
    n_valid = ew_valid.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=ew_pred.device)
    pred_norm = pred[ew_valid] / EW_SCALE
    target_norm = target[ew_valid] / EW_SCALE
    return ((pred_norm - target_norm) ** 2).mean()


def direct_ew_loss(ew_direct, ews, mask):
    """
    Loss for the direct prediction of EW_{t+1}.

    Directly predicts the EW value of the next time step.
    Normalised by EW_SCALE to stabilise the gradient.
    Uses Smooth L1 (Huber) instead of MSE for robustness to
    outliers (some extreme EW values in BeSS).
    """
    valid = mask[:, :-1] * mask[:, 1:]
    pred = ew_direct[:, :-1]
    target = ews[:, 1:]
    not_nan = ~torch.isnan(target)
    ew_valid = (valid > 0) & not_nan
    n_valid = ew_valid.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=ew_direct.device)
    pred_norm = pred[ew_valid] / EW_SCALE
    target_norm = target[ew_valid] / EW_SCALE
    return F.smooth_l1_loss(pred_norm, target_norm)


def contrastive_loss(proj, mask, tau=0.07):
    B, T, D = proj.shape
    valid = mask[:, :-1] * mask[:, 1:]
    valid_flat = valid.reshape(-1)
    anchors = proj[:, :-1].reshape(-1, D)
    positives = proj[:, 1:].reshape(-1, D)
    valid_idx = valid_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) < 2:
        return torch.tensor(0.0, device=proj.device)
    anchors = anchors[valid_idx]
    positives = positives[valid_idx]
    N = len(anchors)
    sim_matrix = anchors @ positives.T / tau
    labels = torch.arange(N, device=proj.device)
    return F.cross_entropy(sim_matrix, labels)
