"""t3_chronos_zeroshot_samepairs.py — Chronos-Bolt-base zero-shot on the exact
populations of the trained baselines.

  (a) Population of ``t3_ts_fm_baselines.py`` (stars with >= 3 finite
      aligned epochs): n = 8,118 pairs, persistence MAE 1.6307 A. Sanity
      check against the reference ratio 0.9676.
  (b) ctx >= 5 subset: n = 7,313 pairs, persistence MAE 1.4723 A, i.e. the
      exact test pairs of PCA(10) ctx=5 and of the trained stacking head
      (``t3_chronos_trained_head.py``). This is the like-for-like zero-shot
      point for those two rows.

Protocol identical to ``t3_ts_fm_baselines.py``: temporal cutoff MJD 59215,
causal contexts of up to 64 past EW values, 1-step ahead, median quantile
point forecast, paired bootstrap CI95 (10,000 resamples).

Output: <BESS_DOWNSTREAM_DIR>/t3_chronos_zeroshot_samepairs.json
"""

import json
import numpy as np
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

HF_ID = "amazon/chronos-bolt-base"
CUTOFF_MJD = 59215.0
MAX_CONTEXT = 64
HEAD_CTX = 5
BATCH = 64
N_BOOT = 10_000


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
        if len(mjds) < 3:  # published load_aligned_data star filter
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


def chronos_predict(contexts):
    from chronos import BaseChronosPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {HF_ID} on {device} (float32)", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        HF_ID, device_map=device, torch_dtype=torch.float32)
    preds = np.empty(len(contexts), dtype=np.float32)
    nb = (len(contexts) + BATCH - 1) // BATCH
    for bi in range(nb):
        s, e = bi * BATCH, min((bi + 1) * BATCH, len(contexts))
        batch = [torch.as_tensor(c, dtype=torch.float32) for c in contexts[s:e]]
        q, _ = pipe.predict_quantiles(inputs=batch, prediction_length=1,
                                      quantile_levels=[0.1, 0.5, 0.9])
        preds[s:e] = q[:, 0, 1].float().cpu().numpy()
        if (bi + 1) % 50 == 0 or bi == nb - 1:
            print(f"  batch {bi+1}/{nb}", flush=True)
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
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
    print(f"population (a): n={len(y)} (expected 8118) | "
          f"persistence MAE {mae_p:.4f} (expected 1.6307)", flush=True)

    preds = chronos_predict(contexts)
    err_m, err_p = np.abs(y - preds), np.abs(y - persist)
    mae = float(err_m.mean())
    res = {"protocol": "published population (>=3-epoch star filter), "
                       "cutoff MJD 59215, context <=64, 1-step, median forecast",
           "full_published_population": {
               "n_test": int(len(y)), "mae_persist": mae_p, "mae": mae,
               "ratio": mae / mae_p, "ci95": ratio_ci(err_m, err_p, rng),
               "published_reference_ratio": 0.9676},
           }
    print(f"(a) full 8118 pairs : ratio {mae/mae_p:.4f} "
          f"(reference 0.9676)", flush=True)

    keep = np.array([c.size >= HEAD_CTX for c in contexts])
    y5, p5 = y[keep], persist[keep]
    m5 = preds[keep]
    mae_p5 = float(np.mean(np.abs(y5 - p5)))
    mae5 = float(np.mean(np.abs(y5 - m5)))
    res["ctx5_subset_same_pairs_as_stacking_head"] = {
        "n_test": int(keep.sum()), "mae_persist": mae_p5, "mae": mae5,
        "ratio": mae5 / mae_p5,
        "ci95": ratio_ci(np.abs(y5 - m5), np.abs(y5 - p5), rng),
        "note": "identical pairs to PCA(10) ctx5 (0.942), trained stacking "
                "head (0.932) and its control (0.945)"}
    print(f"(b) ctx>=5 subset : n={int(keep.sum())} (expected 7313), "
          f"persist {mae_p5:.4f} (expected 1.4723), "
          f"ratio {mae5/mae_p5:.4f}", flush=True)

    with open(f"{OUT}/t3_chronos_zeroshot_samepairs.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/t3_chronos_zeroshot_samepairs.json")


if __name__ == "__main__":
    main()
