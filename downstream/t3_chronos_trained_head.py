"""t3_chronos_trained_head.py — trained linear head on a frozen foundation model.

EWForecast companion experiment to ``t3_ts_fm_baselines.py``: is part of the
gap between the zero-shot foundation models and PCA+ridge due to the fact
that only the latter is trained in-domain?

Same protocol as the zero-shot baselines (temporal cutoff MJD 59215, causal
contexts of up to 64 past EW values, 1-step-ahead), then TRAINED heads:

  ZS   : Chronos-Bolt-base zero-shot median            (reference)
  H1   : ridge on [chronos_pred, ew_{t-1}]             (2-feature calibration)
  H2   : ridge on [chronos_pred, ew_{t-5..t-1}]        (FM forecast + 5-EW context)
  H3   : ridge on [ew_{t-5..t-1}]                      (control: no FM)

The foundation model stays frozen; only the ridge head is fitted, on
pre-cutoff pairs, and evaluated on the post-cutoff pairs. H2/H3 keep the
pairs with >= 5 context points, i.e. exactly the 7,313 test pairs of
PCA(10) ctx=5 (persistence MAE 1.4723 A).

Note on populations: this script keeps stars with >= 2 aligned epochs, so
its ZS/H1 rows are on 8,210 pairs (persistence 1.6532 A) rather than the
8,118 pairs of ``t3_ts_fm_baselines.py`` (>= 3 epochs). The like-for-like
zero-shot number on the 7,313 pairs is produced by
``t3_chronos_zeroshot_samepairs.py``.

Output: <BESS_DOWNSTREAM_DIR>/t3_chronos_trained_head.json
"""

import json
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
HEAD_CTX = 5
BATCH = 64
ALPHAS = np.logspace(-3, 5, 20)
N_BOOT = 10_000
HF_ID = "amazon/chronos-bolt-base"


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
        if len(mjds) < 2:
            continue
        order = np.argsort(mjds)
        series[star] = {"mjds": np.asarray(mjds)[order],
                        "ews": np.asarray(ews, np.float32)[order]}
    return series


def build_pairs(series, side):
    """side='test': targets after cutoff; side='train': targets before."""
    contexts, y, persist = [], [], []
    for star, d in series.items():
        mjds, ews = d["mjds"], d["ews"]
        if side == "test":
            positions = np.where(mjds > CUTOFF_MJD)[0]
        else:
            positions = np.where(mjds <= CUTOFF_MJD)[0]
        for t in positions:
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
        if (bi + 1) % 100 == 0 or bi == nb - 1:
            print(f"  batch {bi+1}/{nb}", flush=True)
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds


def ratio_ci(err_model, err_persist, rng):
    """Paired bootstrap CI95 on MAE(model)/MAE(persist)."""
    n = len(err_model)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = err_model[idx].mean(axis=1) / err_persist[idx].mean(axis=1)
    return [float(v) for v in np.percentile(boots, [2.5, 97.5])]


def main():
    rng = np.random.default_rng(42)
    series = load_series()
    print(f"stars with aligned EW series: {len(series)}")

    ctx_te, y_te, per_te = build_pairs(series, "test")
    ctx_tr, y_tr, per_tr = build_pairs(series, "train")
    print(f"test pairs: {len(y_te)}")
    print(f"train pairs: {len(y_tr)}")
    mae_persist = float(np.mean(np.abs(y_te - per_te)))
    print(f"persistence MAE (test): {mae_persist:.4f}")

    zs_te = chronos_predict(ctx_te)
    zs_tr = chronos_predict(ctx_tr)
    mae_zs = float(np.mean(np.abs(y_te - zs_te)))
    print(f"Chronos-Bolt-base zero-shot MAE: {mae_zs:.4f} "
          f"ratio {mae_zs/mae_persist:.4f}  (reference ratio: 0.968)")

    res = {"n_test": int(len(y_te)), "n_train": int(len(y_tr)),
           "mae_persist": mae_persist,
           "zero_shot": {"mae": mae_zs, "ratio": mae_zs / mae_persist,
                         "ci95": ratio_ci(np.abs(y_te - zs_te),
                                          np.abs(y_te - per_te), rng)}}

    def fit_eval(name, Xtr, ytr, Xte, yte, pte):
        model = Pipeline([("sc", StandardScaler()),
                          ("ridge", RidgeCV(alphas=ALPHAS))])
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = float(np.mean(np.abs(yte - pred)))
        mp = float(np.mean(np.abs(yte - pte)))
        ci = ratio_ci(np.abs(yte - pred), np.abs(yte - pte), rng)
        res[name] = {"mae": mae, "mae_persist_same_pairs": mp,
                     "ratio": mae / mp, "ci95": ci, "n_test": int(len(yte))}
        print(f"{name}: MAE {mae:.4f}  ratio {mae/mp:.4f}  CI95 {ci}")

    # H1: calibration head on [chronos_pred, persistence]
    fit_eval("H1_chronos_plus_last",
             np.stack([zs_tr, per_tr], 1), y_tr,
             np.stack([zs_te, per_te], 1), y_te, per_te)

    # H2/H3: restrict to pairs with >= HEAD_CTX context points
    def last_k(ctxs):
        keep, feats = [], []
        for i, c in enumerate(ctxs):
            if len(c) >= HEAD_CTX:
                keep.append(i)
                feats.append(c[-HEAD_CTX:])
        return np.asarray(keep), np.asarray(feats, np.float32)

    k_tr, f_tr = last_k(ctx_tr)
    k_te, f_te = last_k(ctx_te)
    fit_eval("H2_chronos_plus_ctx5",
             np.concatenate([zs_tr[k_tr, None], f_tr], 1), y_tr[k_tr],
             np.concatenate([zs_te[k_te, None], f_te], 1), y_te[k_te],
             per_te[k_te])
    fit_eval("H3_ctx5_only_control",
             f_tr, y_tr[k_tr], f_te, y_te[k_te], per_te[k_te])

    with open(f"{OUT}/t3_chronos_trained_head.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/t3_chronos_trained_head.json")


if __name__ == "__main__":
    main()
