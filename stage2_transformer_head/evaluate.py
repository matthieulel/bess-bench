"""
evaluate.py — Full evaluation of the Stage 2 Hα temporal model (causal
Transformer head on BeMAE embeddings).

Loads a checkpoint and the config.json written next to it by train.py, then
runs the following evaluations (T0a to T0c on the observations of the test
stars after cfg.cutoff_mjd, T0d on full sequences):
  - T0a) next-embedding prediction. The model/persistence MSE ratio is
    structurally biased when the Hα embeddings are stable, so metrics that
    measure the ability of the model to predict CHANGES are also reported:
    • change_mse : MSE only on the steps where z changes significantly
    • delta_cos  : cosine similarity between Δz_pred and Δz_true
    • surprise_corr : correlation between the model surprise and the
      amplitude of the change
    Results are also given per category of number of observations per star.
  - T0b) EW tracking via a post-hoc Ridge regression z → EW
  - T0b_direct) EW tracking with the direct EW head (no Ridge needed)
  - T0c) and T0c_direct) trend classification (sign of ΔEW)
  - T0d) change-point detection (gold events and data-driven EW jumps)

The results are printed and saved as JSON (--output, default:
eval_results.json next to the checkpoint).

Usage:
    cd stage2_transformer_head
    python evaluate.py --checkpoint runs/base/best.pt
    python evaluate.py --checkpoint runs/base/best.pt --output eval_base.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

from config import TemporalModelConfig, TemporalTrainConfig
from dataset import load_star_data
from model import TemporalAggregator, EW_SCALE


# ══════════════════════════════════════════════════════════════════════════════
# GOLD EVENTS
# ══════════════════════════════════════════════════════════════════════════════

GOLD_EVENTS = {
    "DEL SCO": [
        {"mjd": 51700, "type": "disk_onset", "desc": "Disk onset 2000 (B → Be)"},
    ],
    "PLEIONE": [
        {"mjd": 53500, "type": "transition", "desc": "Shell → Be around 2005"},
    ],
    "PI AQR": [
        {"mjd": 51600, "type": "outburst", "desc": "Outburst 2000"},
        {"mjd": 52500, "type": "outburst", "desc": "Outburst 2002"},
    ],
    "ZET TAU": [
        {"mjd": 50500, "type": "vr_reversal", "desc": "V/R reversal ~1997"},
        {"mjd": 51900, "type": "vr_reversal", "desc": "V/R reversal ~2001"},
    ],
    "GAM CAS": [
        {"mjd": 53000, "type": "activity", "desc": "Sustained activity 2003-2004"},
    ],
    "BK CAM": [
        {"mjd": 55000, "type": "outburst", "desc": "Outburst ~2009"},
    ],
    "ETA CEN": [
        {"mjd": 50200, "type": "outburst", "desc": "Outburst ~1996"},
    ],
}

EVENT_TOLERANCE_MJD = 180


# ══════════════════════════════════════════════════════════════════════════════
# T0a) NEXT-EMBEDDING PREDICTION (WITH CHANGE METRICS)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_next_embedding(model, star_data, test_stars, cfg, device):
    """
    Evaluates the next-embedding prediction.

    In addition to the usual MSE ratio (structurally biased for Hα),
    CHANGE metrics are reported:
      - change_mse : MSE only on the steps with a large Δz
      - delta_cos  : cos(Δz_pred, Δz_true) — does the model predict the
                     right direction of change?
      - surprise_corr : correlation between the surprise and the actual
                        amplitude of the change
    """
    model.eval()
    results_per_star = {}
    all_model_mse = []
    all_persist_mse = []
    all_mean_mse = []
    all_cos_sim = []

    # Change metrics
    all_delta_cos = []       # cos(Δz_pred, Δz_true)
    all_change_mse = []      # MSE on the steps with a large change
    all_change_persist = []  # Persistence MSE on the same steps
    all_surprise = []        # model surprise
    all_actual_change = []   # actual amplitude of the change

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        ews = data["ews"].numpy()

        train_mask = mjds <= cfg.cutoff_mjd
        test_mask = mjds > cfg.cutoff_mjd
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) < 3 or len(test_idx) < 2:
            continue

        global_mean = embeddings[train_idx].mean(dim=0)

        star_model_mse = []
        star_persist_mse = []
        star_mean_mse = []
        star_cos = []
        star_delta_cos = []

        for i, t_idx in enumerate(test_idx[:-1]):
            # Context: train + test points already seen, INCLUDING t_idx
            all_ctx_idx = np.concatenate([train_idx, test_idx[:i + 1]])

            if len(all_ctx_idx) > cfg.max_seq_len:
                all_ctx_idx = all_ctx_idx[-cfg.max_seq_len:]

            ctx_e = embeddings[all_ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[all_ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask_t = torch.ones(1, len(all_ctx_idx), device=device)

            out = model(ctx_e, ctx_m, ctx_mask_t)
            z_pred = out["z_pred"][0, -1]
            z_target = embeddings[test_idx[i + 1]].to(device)
            z_current = embeddings[t_idx].to(device)

            mse_model = float(((z_pred - z_target) ** 2).mean())
            mse_persist = float(((z_current - z_target) ** 2).mean())
            mse_mean = float(((global_mean.to(device) - z_target) ** 2).mean())
            cos = float(F.cosine_similarity(
                z_pred.unsqueeze(0), z_target.unsqueeze(0)
            ))

            star_model_mse.append(mse_model)
            star_persist_mse.append(mse_persist)
            star_mean_mse.append(mse_mean)
            star_cos.append(cos)

            # Change metrics
            delta_real = z_target - z_current   # actual change
            delta_pred = z_pred - z_current     # predicted change
            actual_change = float(delta_real.norm())

            if actual_change > 1e-6:
                dc = float(F.cosine_similarity(
                    delta_pred.unsqueeze(0), delta_real.unsqueeze(0)
                ))
                star_delta_cos.append(dc)
                all_delta_cos.append(dc)

            all_surprise.append(mse_model)
            all_actual_change.append(actual_change)

            # Change MSE: only on the steps with a large change
            if mse_persist > 0.005:  # threshold for a "significant change"
                all_change_mse.append(mse_model)
                all_change_persist.append(mse_persist)

        if star_model_mse:
            results_per_star[star] = {
                "n_test": len(test_idx),
                "model_mse": float(np.mean(star_model_mse)),
                "persist_mse": float(np.mean(star_persist_mse)),
                "mean_mse": float(np.mean(star_mean_mse)),
                "cos_sim": float(np.mean(star_cos)),
                "ratio_vs_persist": (
                    float(np.mean(star_model_mse)) /
                    max(float(np.mean(star_persist_mse)), 1e-10)
                ),
                "delta_cos": float(np.mean(star_delta_cos)) if star_delta_cos else None,
            }
            all_model_mse.extend(star_model_mse)
            all_persist_mse.extend(star_persist_mse)
            all_mean_mse.extend(star_mean_mse)
            all_cos_sim.extend(star_cos)

    # Global summary
    summary = {
        "n_stars": len(results_per_star),
        "n_predictions": len(all_model_mse),
        "model_mse": float(np.mean(all_model_mse)) if all_model_mse else 0.0,
        "persist_mse": float(np.mean(all_persist_mse)) if all_persist_mse else 0.0,
        "mean_mse": float(np.mean(all_mean_mse)) if all_mean_mse else 0.0,
        "cos_sim": float(np.mean(all_cos_sim)) if all_cos_sim else 0.0,
    }

    if summary["persist_mse"] > 0:
        summary["ratio_vs_persist"] = summary["model_mse"] / summary["persist_mse"]
    else:
        summary["ratio_vs_persist"] = float("inf")

    n_better = sum(1 for s in results_per_star.values()
                   if s["ratio_vs_persist"] < 1.0)
    summary["stars_beating_persist"] = n_better
    summary["pct_beating_persist"] = n_better / max(len(results_per_star), 1) * 100

    # ── Change metrics ──
    summary["delta_cos_mean"] = (
        float(np.mean(all_delta_cos)) if all_delta_cos else 0.0
    )
    summary["delta_cos_median"] = (
        float(np.median(all_delta_cos)) if all_delta_cos else 0.0
    )
    summary["n_delta_cos"] = len(all_delta_cos)

    if all_change_mse:
        summary["change_model_mse"] = float(np.mean(all_change_mse))
        summary["change_persist_mse"] = float(np.mean(all_change_persist))
        summary["change_ratio"] = (
            float(np.mean(all_change_mse)) /
            max(float(np.mean(all_change_persist)), 1e-10)
        )
        summary["n_change_steps"] = len(all_change_mse)
    else:
        summary["change_model_mse"] = None
        summary["change_persist_mse"] = None
        summary["change_ratio"] = None
        summary["n_change_steps"] = 0

    if len(all_surprise) > 10 and len(all_actual_change) > 10:
        from scipy.stats import pearsonr
        corr, _ = pearsonr(all_surprise, all_actual_change)
        summary["surprise_change_corr"] = float(corr)
    else:
        summary["surprise_change_corr"] = 0.0

    # Categories by number of observations per star (from min_obs=5 upwards)
    categories = {
        "A_100plus": {"min": 100, "max": float("inf"), "label": "≥100 obs"},
        "B_30_99":   {"min": 30,  "max": 99,           "label": "30-99 obs"},
        "C_15_29":   {"min": 15,  "max": 29,           "label": "15-29 obs"},
        "D_5_14":    {"min": 5,   "max": 14,           "label": "5-14 obs"},
    }
    by_category = {}
    for cat_name, cat_bounds in categories.items():
        cat_stars = {
            s: r for s, r in results_per_star.items()
            if cat_bounds["min"] <= star_data[s]["mjds"].shape[0] <= cat_bounds["max"]
        }
        if not cat_stars:
            by_category[cat_name] = {"n_stars": 0, "label": cat_bounds["label"]}
            continue
        cat_ratios = [r["ratio_vs_persist"] for r in cat_stars.values()]
        cat_better = sum(1 for r in cat_ratios if r < 1.0)
        cat_delta_cos = [r["delta_cos"] for r in cat_stars.values()
                         if r["delta_cos"] is not None]
        by_category[cat_name] = {
            "label": cat_bounds["label"],
            "n_stars": len(cat_stars),
            "ratio_vs_persist": float(np.mean(cat_ratios)),
            "stars_beating_persist": cat_better,
            "pct_beating_persist": cat_better / len(cat_stars) * 100,
            "delta_cos_mean": float(np.mean(cat_delta_cos)) if cat_delta_cos else None,
        }
    summary["by_category"] = by_category

    return summary, results_per_star


# ══════════════════════════════════════════════════════════════════════════════
# T0b) EW(Hα) TRACKING (post-hoc Ridge)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ew_tracking(model, star_data, test_stars, cfg, device):
    """
    Evaluates EW tracking via a post-hoc Ridge regression z→EW.
    The Ridge is fitted on the pre-cutoff observations of the test stars
    and applied to the predicted embedding (model) and to the current
    embedding (persistence).
    """
    train_z = []
    train_ew = []

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        ews = data["ews"].numpy()
        train_mask = mjds <= cfg.cutoff_mjd
        for i in np.where(train_mask)[0]:
            if not np.isnan(ews[i]):
                train_z.append(data["embeddings"][i].numpy())
                train_ew.append(ews[i])

    if len(train_z) < 10:
        return {"error": "Not enough EW data for the regression"}, {}

    train_z = np.stack(train_z)
    train_ew = np.clip(np.array(train_ew), -100, 100)

    reg = Ridge(alpha=10.0)
    reg.fit(train_z, train_ew)

    all_ew_true = []
    all_ew_pred_model = []
    all_ew_pred_persist = []
    model.eval()

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        ews = data["ews"].numpy()
        train_idx = np.where(mjds <= cfg.cutoff_mjd)[0]
        test_idx = np.where(mjds > cfg.cutoff_mjd)[0]

        if len(train_idx) < 3 or len(test_idx) < 2:
            continue

        for i, t_idx in enumerate(test_idx[:-1]):
            ew_target = ews[test_idx[i + 1]]
            if np.isnan(ew_target):
                continue

            # Context: train + test points already seen, INCLUDING t_idx
            all_ctx_idx = np.concatenate([train_idx, test_idx[:i + 1]])

            if len(all_ctx_idx) > cfg.max_seq_len:
                all_ctx_idx = all_ctx_idx[-cfg.max_seq_len:]

            ctx_e = embeddings[all_ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[all_ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(all_ctx_idx), device=device)

            out = model(ctx_e, ctx_m, ctx_mask)
            z_pred = out["z_pred"][0, -1].cpu().numpy()
            z_persist = embeddings[t_idx].numpy()

            all_ew_true.append(np.clip(ew_target, -100, 100))
            all_ew_pred_model.append(float(reg.predict(z_pred.reshape(1, -1))[0]))
            all_ew_pred_persist.append(float(reg.predict(z_persist.reshape(1, -1))[0]))

    if len(all_ew_true) < 5:
        return {"error": "Not enough valid EW predictions"}, {}

    ew_true = np.array(all_ew_true)
    ew_model = np.array(all_ew_pred_model)
    ew_persist = np.array(all_ew_pred_persist)

    from scipy.stats import pearsonr

    ss_res = np.sum((ew_model - ew_true) ** 2)
    ss_tot = np.sum((ew_true - ew_true.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {
        "n_predictions": len(ew_true),
        "mae_model": float(np.mean(np.abs(ew_model - ew_true))),
        "mae_persist": float(np.mean(np.abs(ew_persist - ew_true))),
        "corr_model": float(pearsonr(ew_true, ew_model)[0]) if len(ew_true) > 2 else 0.0,
        "corr_persist": float(pearsonr(ew_true, ew_persist)[0]) if len(ew_true) > 2 else 0.0,
        "r2_model": float(r2),
        "ratio_mae": float(np.mean(np.abs(ew_model - ew_true))) /
                     max(float(np.mean(np.abs(ew_persist - ew_true))), 1e-10),
    }, {}


# ══════════════════════════════════════════════════════════════════════════════
# T0b_direct) EW(Hα) TRACKING — DIRECT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ew_direct(model, star_data, test_stars, cfg, device):
    """
    Evaluates the direct prediction of EW_{t+1} by the DirectEWHead head.
    No post-hoc Ridge needed — the model predicts EW directly.
    """
    model.eval()
    all_ew_true = []
    all_ew_pred = []
    all_ew_persist = []  # persistence baseline: EW_{t+1} = EW_{t}

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        ews = data["ews"].numpy()
        train_idx = np.where(mjds <= cfg.cutoff_mjd)[0]
        test_idx = np.where(mjds > cfg.cutoff_mjd)[0]

        if len(train_idx) < 3 or len(test_idx) < 2:
            continue

        for i, t_idx in enumerate(test_idx[:-1]):
            ew_target = ews[test_idx[i + 1]]
            ew_current = ews[t_idx]
            if np.isnan(ew_target) or np.isnan(ew_current):
                continue

            # Context: train + test points already seen, INCLUDING t_idx
            all_ctx_idx = np.concatenate([train_idx, test_idx[:i + 1]])

            if len(all_ctx_idx) > cfg.max_seq_len:
                all_ctx_idx = all_ctx_idx[-cfg.max_seq_len:]

            ctx_e = embeddings[all_ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[all_ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(all_ctx_idx), device=device)

            out = model(ctx_e, ctx_m, ctx_mask)
            ew_pred = float(out["ew_direct"][0, -1].cpu())

            all_ew_true.append(np.clip(ew_target, -100, 100))
            all_ew_pred.append(ew_pred)
            all_ew_persist.append(np.clip(ew_current, -100, 100))

    if len(all_ew_true) < 5:
        return {"error": "Not enough valid direct EW predictions"}

    ew_true = np.array(all_ew_true)
    ew_pred = np.array(all_ew_pred)
    ew_persist = np.array(all_ew_persist)

    from scipy.stats import pearsonr

    mae_model = float(np.mean(np.abs(ew_pred - ew_true)))
    mae_persist = float(np.mean(np.abs(ew_persist - ew_true)))

    ss_res = np.sum((ew_pred - ew_true) ** 2)
    ss_tot = np.sum((ew_true - ew_true.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    corr = float(pearsonr(ew_true, ew_pred)[0]) if len(ew_true) > 2 else 0.0

    return {
        "n_predictions": len(ew_true),
        "mae_model": mae_model,
        "mae_persist": mae_persist,
        "r2_model": float(r2),
        "corr_model": corr,
        "ratio_mae": mae_model / max(mae_persist, 1e-10),
    }


# ══════════════════════════════════════════════════════════════════════════════
# T0c) TREND CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_trend(model, star_data, test_stars, cfg, device):
    """
    Evaluates the classification of the activity trend (sign of ΔEW),
    with EW obtained from the predicted embedding through the post-hoc
    Ridge z→EW.
    """
    model.eval()
    train_z = []
    train_ew = []

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        ews = data["ews"].numpy()
        train_mask = mjds <= cfg.cutoff_mjd
        for i in np.where(train_mask)[0]:
            if not np.isnan(ews[i]):
                train_z.append(data["embeddings"][i].numpy())
                train_ew.append(ews[i])

    if len(train_z) < 10:
        return {"error": "Not enough data for the EW regression"}, {}

    reg = Ridge(alpha=10.0)
    reg.fit(np.stack(train_z), np.clip(np.array(train_ew), -100, 100))

    true_trends = []
    pred_trends_model = []
    pred_trends_persist = []

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        ews = data["ews"].numpy()
        train_idx = np.where(mjds <= cfg.cutoff_mjd)[0]
        test_idx = np.where(mjds > cfg.cutoff_mjd)[0]

        if len(train_idx) < 3 or len(test_idx) < 3:
            continue

        for i in range(1, len(test_idx) - 1):
            ew_current = ews[test_idx[i]]
            ew_next = ews[test_idx[i + 1]]
            if np.isnan(ew_current) or np.isnan(ew_next):
                continue

            true_trend = 1 if (ew_next - ew_current) < 0 else 0

            all_ctx_idx = np.concatenate([train_idx, test_idx[:i + 1]])
            if len(all_ctx_idx) > cfg.max_seq_len:
                all_ctx_idx = all_ctx_idx[-cfg.max_seq_len:]

            ctx_e = embeddings[all_ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[all_ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(all_ctx_idx), device=device)

            out = model(ctx_e, ctx_m, ctx_mask)
            z_pred = out["z_pred"][0, -1].cpu().numpy()

            ew_pred_model = float(reg.predict(z_pred.reshape(1, -1))[0])
            ew_pred_persist = float(reg.predict(
                embeddings[test_idx[i]].numpy().reshape(1, -1)
            )[0])

            pred_trends_model.append(1 if (ew_pred_model - ew_current) < 0 else 0)
            pred_trends_persist.append(1 if (ew_pred_persist - ew_current) < 0 else 0)
            true_trends.append(true_trend)

    if len(true_trends) < 10:
        return {"error": "Too few trends to evaluate"}, {}

    true_arr = np.array(true_trends)
    pred_m = np.array(pred_trends_model)
    pred_p = np.array(pred_trends_persist)

    return {
        "n_predictions": len(true_arr),
        "accuracy_model": float(accuracy_score(true_arr, pred_m)),
        "accuracy_persist": float(accuracy_score(true_arr, pred_p)),
        "f1_model": float(f1_score(true_arr, pred_m, zero_division=0)),
        "f1_persist": float(f1_score(true_arr, pred_p, zero_division=0)),
        "prevalence_up": float(true_arr.mean()),
    }, {}


# ══════════════════════════════════════════════════════════════════════════════
# T0c_direct) TREND VIA DIRECT EW PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_trend_direct(model, star_data, test_stars, cfg, device):
    """
    Trend via the direct EW prediction — no Ridge.
    """
    model.eval()
    true_trends = []
    pred_trends = []

    for star in test_stars:
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        ews = data["ews"].numpy()
        train_idx = np.where(mjds <= cfg.cutoff_mjd)[0]
        test_idx = np.where(mjds > cfg.cutoff_mjd)[0]

        if len(train_idx) < 3 or len(test_idx) < 3:
            continue

        for i in range(1, len(test_idx) - 1):
            ew_current = ews[test_idx[i]]
            ew_next = ews[test_idx[i + 1]]
            if np.isnan(ew_current) or np.isnan(ew_next):
                continue

            true_trend = 1 if (ew_next - ew_current) < 0 else 0

            all_ctx_idx = np.concatenate([train_idx, test_idx[:i + 1]])
            if len(all_ctx_idx) > cfg.max_seq_len:
                all_ctx_idx = all_ctx_idx[-cfg.max_seq_len:]

            ctx_e = embeddings[all_ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[all_ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(all_ctx_idx), device=device)

            out = model(ctx_e, ctx_m, ctx_mask)
            ew_pred = float(out["ew_direct"][0, -1].cpu())

            pred_trends.append(1 if (ew_pred - ew_current) < 0 else 0)
            true_trends.append(true_trend)

    if len(true_trends) < 10:
        return {"error": "Too few trends to evaluate"}

    true_arr = np.array(true_trends)
    pred_arr = np.array(pred_trends)

    return {
        "n_predictions": len(true_arr),
        "accuracy": float(accuracy_score(true_arr, pred_arr)),
        "f1": float(f1_score(true_arr, pred_arr, zero_division=0)),
        "prevalence_up": float(true_arr.mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# T0d) CHANGE-POINT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def discover_ew_events(star_data, threshold_sigma=3.0, min_obs=30, top_k=5):
    """Automatically discovers events (change-points) from EW jumps."""
    discovered = {}
    for star, data in star_data.items():
        ews = data["ews"].numpy()
        mjds = data["mjds"].numpy()
        valid = ~np.isnan(ews)
        if valid.sum() < min_obs:
            continue
        ews_v = ews[valid]
        mjds_v = mjds[valid]
        delta = np.abs(np.diff(ews_v))
        if len(delta) < 10:
            continue
        mu, sigma = np.mean(delta), np.std(delta)
        threshold = mu + threshold_sigma * sigma
        events = []
        for i, d in enumerate(delta):
            if d > threshold:
                events.append({
                    "mjd": float(mjds_v[i + 1]),
                    "type": "ew_jump",
                    "delta_ew": float(np.diff(ews_v)[i]),
                })
        if events:
            events.sort(key=lambda e: abs(e.get("delta_ew", 0)), reverse=True)
            discovered[star] = events[:top_k]
    return discovered


@torch.no_grad()
def evaluate_change_points(model, star_data, test_stars, cfg, device):
    """Evaluates the detection of known (gold) and data-driven events."""
    model.eval()
    results = {}
    total_events = 0
    detected_events = 0
    event_details = []

    for star in star_data:
        if star not in GOLD_EVENTS:
            continue
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        if len(mjds) < 10:
            continue

        surprises = []
        for i in range(1, len(mjds)):
            ctx_idx = list(range(max(0, i - cfg.max_seq_len), i))
            ctx_e = embeddings[ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(ctx_idx), device=device)
            out = model(ctx_e, ctx_m, ctx_mask)
            z_pred = out["z_pred"][0, -1]
            z_real = embeddings[i].to(device)
            mse = float(((z_pred - z_real) ** 2).mean())
            surprises.append({"mjd": float(mjds[i]), "surprise": mse})

        if not surprises:
            continue

        s_values = np.array([s["surprise"] for s in surprises])
        threshold = s_values.mean() + 2 * s_values.std()
        peak_mjds = [s["mjd"] for s in surprises if s["surprise"] > threshold]

        for event in GOLD_EVENTS[star]:
            total_events += 1
            event_mjd = event["mjd"]
            obs_gaps = np.abs(np.array([s["mjd"] for s in surprises]) - event_mjd)
            closest_idx = int(np.argmin(obs_gaps))
            closest_gap = float(obs_gaps[closest_idx])
            closest_surprise = surprises[closest_idx]["surprise"]
            surprise_rank = float(np.mean(s_values >= closest_surprise))

            detected = any(abs(p - event_mjd) <= EVENT_TOLERANCE_MJD for p in peak_mjds)
            if detected:
                detected_events += 1

            event_details.append({
                "star": star, "event_mjd": event_mjd,
                "event_type": event["type"], "detected": detected,
                "closest_obs_gap_days": closest_gap,
                "closest_surprise_rank": surprise_rank,
                "has_obs_in_window": closest_gap <= EVENT_TOLERANCE_MJD,
            })

        results[star] = {
            "n_events": len(GOLD_EVENTS[star]),
            "n_detected": sum(1 for ed in event_details
                            if ed["star"] == star and ed["detected"]),
            "n_peaks": len(peak_mjds),
        }

    # Data-driven
    data_driven = discover_ew_events(star_data)
    dd_total = 0
    dd_detected = 0

    for star, dd_events in data_driven.items():
        data = star_data[star]
        mjds = data["mjds"].numpy()
        embeddings = data["embeddings"]
        if len(mjds) < 10:
            continue

        s_vals = []
        s_mjds = []
        for i in range(1, len(mjds)):
            ctx_idx = list(range(max(0, i - cfg.max_seq_len), i))
            ctx_e = embeddings[ctx_idx].unsqueeze(0).to(device)
            ctx_m = torch.tensor(
                mjds[ctx_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            ctx_mask = torch.ones(1, len(ctx_idx), device=device)
            out = model(ctx_e, ctx_m, ctx_mask)
            z_pred = out["z_pred"][0, -1]
            z_real = embeddings[i].to(device)
            s_vals.append(float(((z_pred - z_real) ** 2).mean()))
            s_mjds.append(float(mjds[i]))

        if not s_vals:
            continue

        s_arr = np.array(s_vals)
        s_mjds_arr = np.array(s_mjds)
        threshold = s_arr.mean() + 2 * s_arr.std()

        for ev in dd_events:
            dd_total += 1
            idx_match = np.argmin(np.abs(s_mjds_arr - ev["mjd"]))
            if s_arr[idx_match] > threshold:
                dd_detected += 1

    summary = {
        "total_events": total_events,
        "detected_events": detected_events,
        "detection_rate": detected_events / max(total_events, 1),
        "stars_evaluated": len(results),
        "event_details": event_details,
        "dd_total_events": dd_total,
        "dd_detected_events": dd_detected,
        "dd_detection_rate": dd_detected / max(dd_total, 1),
        "dd_stars_evaluated": len(data_driven),
    }
    return summary, results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(checkpoint_path: str, output_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(checkpoint_path).parent
    with open(ckpt_dir / "config.json") as f:
        config = json.load(f)

    model_cfg = TemporalModelConfig(**config["model"])
    train_cfg = TemporalTrainConfig(**config["train"])

    model = TemporalAggregator(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Model loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    star_data, split_info = load_star_data(train_cfg)
    test_stars = split_info["test_stars"]
    print(f"Test set: {len(test_stars)} stars")

    results = {}

    # ── T0a ──
    print("\n" + "=" * 60)
    print("T0a) Next-embedding prediction (with change metrics)")
    print("=" * 60)
    s_ne, _ = evaluate_next_embedding(model, star_data, test_stars, train_cfg, device)
    results["T0a_next_embedding"] = s_ne
    print(f"  MSE model        : {s_ne['model_mse']:.6f}")
    print(f"  MSE persistence  : {s_ne['persist_mse']:.6f}")
    print(f"  Ratio vs pers.   : {s_ne['ratio_vs_persist']:.3f}")
    print(f"  Cosine sim       : {s_ne['cos_sim']:.3f}")
    print(f"  Stars > pers.    : {s_ne['stars_beating_persist']}/{s_ne['n_stars']} "
          f"({s_ne['pct_beating_persist']:.1f}%)")
    print(f"  ── Change metrics ──")
    print(f"  Delta cos mean   : {s_ne['delta_cos_mean']:.3f} (n={s_ne['n_delta_cos']})")
    print(f"  Delta cos median : {s_ne['delta_cos_median']:.3f}")
    if s_ne["n_change_steps"] > 0:
        print(f"  Change ratio     : {s_ne['change_ratio']:.3f} "
              f"(n={s_ne['n_change_steps']} steps with a large change)")
    print(f"  Surprise↔change  : r={s_ne['surprise_change_corr']:.3f}")
    if "by_category" in s_ne:
        print("  ── By category ──")
        for cat_name, cat in s_ne["by_category"].items():
            if cat["n_stars"] == 0:
                print(f"    {cat['label']}: 0 stars")
            else:
                dc = f", δcos={cat['delta_cos_mean']:.3f}" if cat.get("delta_cos_mean") else ""
                print(f"    {cat['label']}: ratio={cat['ratio_vs_persist']:.3f}, "
                      f"{cat['stars_beating_persist']}/{cat['n_stars']} > pers.{dc}")

    # ── T0b (Ridge) ──
    print("\n" + "=" * 60)
    print("T0b) EW tracking (Ridge post-hoc)")
    print("=" * 60)
    s_ew, _ = evaluate_ew_tracking(model, star_data, test_stars, train_cfg, device)
    results["T0b_ew_tracking"] = s_ew
    if "error" not in s_ew:
        print(f"  MAE model   : {s_ew['mae_model']:.4f}")
        print(f"  MAE persist : {s_ew['mae_persist']:.4f}")
        print(f"  R² model    : {s_ew['r2_model']:.3f}")
        print(f"  Corr model  : {s_ew['corr_model']:.3f}")
    else:
        print(f"  {s_ew['error']}")

    # ── T0b_direct ──
    print("\n" + "=" * 60)
    print("T0b_direct) EW tracking (direct prediction)")
    print("=" * 60)
    s_ewd = evaluate_ew_direct(model, star_data, test_stars, train_cfg, device)
    results["T0b_direct_ew"] = s_ewd
    if "error" not in s_ewd:
        print(f"  MAE model   : {s_ewd['mae_model']:.4f}")
        print(f"  MAE persist : {s_ewd['mae_persist']:.4f}")
        print(f"  R² model    : {s_ewd['r2_model']:.3f}")
        print(f"  Corr model  : {s_ewd['corr_model']:.3f}")
        print(f"  Ratio MAE   : {s_ewd['ratio_mae']:.3f}")
    else:
        print(f"  {s_ewd['error']}")

    # ── T0c (Ridge) ──
    print("\n" + "=" * 60)
    print("T0c) Trend classification")
    print("=" * 60)
    s_trend, _ = evaluate_trend(model, star_data, test_stars, train_cfg, device)
    results["T0c_trend"] = s_trend
    if "error" not in s_trend:
        print(f"  Accuracy model      : {s_trend['accuracy_model']:.3f}")
        print(f"  Accuracy persistence: {s_trend['accuracy_persist']:.3f}")
        print(f"  F1 model            : {s_trend['f1_model']:.3f}")
    else:
        print(f"  {s_trend['error']}")

    # ── T0c_direct ──
    print("\n" + "=" * 60)
    print("T0c_direct) Direct trend (direct EW prediction)")
    print("=" * 60)
    s_td = evaluate_trend_direct(model, star_data, test_stars, train_cfg, device)
    results["T0c_direct_trend"] = s_td
    if "error" not in s_td:
        print(f"  Accuracy : {s_td['accuracy']:.3f}")
        print(f"  F1       : {s_td['f1']:.3f}")
    else:
        print(f"  {s_td['error']}")

    # ── T0d ──
    print("\n" + "=" * 60)
    print("T0d) Change-point detection")
    print("=" * 60)
    s_cp, details_cp = evaluate_change_points(
        model, star_data, test_stars, train_cfg, device
    )
    results["T0d_change_points"] = s_cp
    print(f"  ── Gold events ──")
    print(f"  Detected : {s_cp['detected_events']}/{s_cp['total_events']} "
          f"({s_cp['detection_rate']:.1%})")
    if "event_details" in s_cp:
        for ed in s_cp["event_details"]:
            status = "✓" if ed["detected"] else "✗"
            obs = "obs OK" if ed["has_obs_in_window"] else f"gap={ed['closest_obs_gap_days']:.0f}j"
            print(f"    {status} {ed['star']} MJD {ed['event_mjd']:.0f} "
                  f"({ed['event_type']}): {obs}")
    print(f"  ── Data-driven (>3σ) ──")
    print(f"  Detected : {s_cp['dd_detected_events']}/{s_cp['dd_total_events']} "
          f"({s_cp['dd_detection_rate']:.1%})")

    # ── Save ──
    if output_path is None:
        output_path = str(ckpt_dir / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Hα — Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output)
