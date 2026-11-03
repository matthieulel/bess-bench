"""t3_moiraimoe_zeroshot.py — Moirai-MoE-1.0-R zero-shot on EWForecast
(Liu et al. 2025; sparse mixture of experts, 32 experts, top-2 routing).

Same protocol and population as ``t3_ts_fm_baselines.py`` and
``t3_moirai_zeroshot.py``: temporal cutoff MJD 59215, causal contexts of up
to 64 past EW(Halpha) values, stars with >= 3 finite aligned epochs,
1-step-ahead forecast, MAE ratio against persistence on the identical
target pairs (n = 8,118, persistence MAE 1.6307 A), paired bootstrap CI95
(10,000 resamples).

Models: Salesforce/moirai-moe-1.0-R-small (117M parameters in total, 11M
activated per token) and -base (935M / 86M); point forecast = median of 100
samples (torch/numpy seeds fixed at 42).

Environment: ``uni2ts`` pins its own torch/gluonts versions; install it in
a separate virtual environment (``pip install uni2ts``), see REPRODUCE.md.
Checkpoints are released by their authors under CC-BY-NC-4.0.

Output: <BESS_DOWNSTREAM_DIR>/t3_moiraimoe_zeroshot.json
"""

import json
import numpy as np
import pandas as pd
import torch

import os
from pathlib import Path

# Repository-relative defaults; override with the same environment variables
# as the other scripts (BESS_EMB_PATH, BESS_FEATURES_PATH, BESS_DOWNSTREAM_DIR).
_BENCH_ROOT = Path(__file__).resolve().parents[1]
EMB = os.environ.get("BESS_EMB_PATH", str(
    _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"))
FEAT = os.environ.get("BESS_FEATURES_PATH", str(
    _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"))
OUT = os.environ.get("BESS_DOWNSTREAM_DIR", str(
    _BENCH_ROOT / "reference_results" / "downstream"))

CUTOFF_MJD = 59215.0
MAX_CONTEXT = 64
NUM_SAMPLES = 100
BATCH = 64
N_BOOT = 10_000
SIZES = ["small", "base"]


def load_series():
    emb = torch.load(EMB, map_location="cpu", weights_only=False)
    feat = torch.load(FEAT, map_location="cpu", weights_only=False)
    series = {}
    for star in sorted(set(emb) & set(feat)):
        es, fs = emb[star], feat[star]
        emjds = es["mjds"].numpy()
        fmjds = fs["mjds"].numpy()
        mjds, ews = [], []
        for m in emjds:
            d = np.abs(fmjds - m)
            b = int(np.argmin(d))
            if d[b] < 0.01:
                ew = float(fs["ew"][b])
                if np.isfinite(ew):
                    mjds.append(float(m))
                    ews.append(ew)
        if len(mjds) < 3:
            continue
        order = np.argsort(mjds)
        series[star] = {"mjds": np.asarray(mjds)[order],
                        "ews": np.asarray(ews, np.float32)[order]}
    return series


def build_test_pairs(series):
    contexts, y, persist = [], [], []
    for star, d in series.items():
        mjds, ews = d["mjds"], d["ews"]
        for t in np.where(mjds > CUTOFF_MJD)[0]:
            if t == 0:
                continue
            ctx = ews[max(0, t - MAX_CONTEXT):t]
            if ctx.size < 1 or not np.all(np.isfinite(ctx)):
                continue
            if not np.isfinite(ews[t]):
                continue
            contexts.append(ctx.astype(np.float32))
            y.append(float(ews[t]))
            persist.append(float(ews[t - 1]))
    return contexts, np.asarray(y), np.asarray(persist)


def moirai_predict(size, contexts):
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

    torch.manual_seed(42)
    np.random.seed(42)
    hf_id = f"Salesforce/moirai-moe-1.0-R-{size}"
    print(f"loading {hf_id} ...", flush=True)
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(hf_id),
        prediction_length=1,
        context_length=MAX_CONTEXT,
        patch_size=16,
        num_samples=NUM_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = model.create_predictor(batch_size=BATCH, device=device)

    start = pd.Period("2000-01-01", freq="D")
    dataset = ListDataset(
        [{"target": c, "start": start} for c in contexts], freq="D")

    preds = np.empty(len(contexts), dtype=np.float32)
    for i, fc in enumerate(predictor.predict(dataset)):
        preds[i] = float(np.median(fc.samples[:, 0]))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(contexts)}", flush=True)
    return preds


def ratio_ci(err_m, err_p, rng):
    n = len(err_m)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = err_m[idx].mean(axis=1) / err_p[idx].mean(axis=1)
    return [float(v) for v in np.percentile(boots, [2.5, 97.5])]


def main():
    rng = np.random.default_rng(42)
    series = load_series()
    contexts, y, persist = build_test_pairs(series)
    mae_p = float(np.mean(np.abs(y - persist)))
    print(f"test pairs: {len(y)} (expected 8118) | "
          f"persistence MAE {mae_p:.4f} (expected ~1.6307)", flush=True)

    res = {"n_test": int(len(y)), "mae_persist": mae_p,
           "protocol": "cutoff MJD 59215, context <=64, 1-step, "
                       "median of 100 samples, seeds 42",
           "published_chronos_bolt_base_same_pairs": 0.9676}
    for size in SIZES:
        preds = moirai_predict(size, contexts)
        mae = float(np.mean(np.abs(y - preds)))
        ci = ratio_ci(np.abs(y - preds), np.abs(y - persist), rng)
        res[f"moirai_moe_1.0_R_{size}"] = {
            "params_M_total": {"small": 117, "base": 935}[size],
            "params_M_activated": {"small": 11, "base": 86}[size],
            "mae": mae, "ratio": mae / mae_p, "ci95": ci}
        print(f"Moirai-MoE-1.0-R-{size}: MAE {mae:.4f}  ratio {mae/mae_p:.4f}  "
              f"CI95 [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)

    with open(f"{OUT}/t3_moiraimoe_zeroshot.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/t3_moiraimoe_zeroshot.json")


if __name__ == "__main__":
    main()
