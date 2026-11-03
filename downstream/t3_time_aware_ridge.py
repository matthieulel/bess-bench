"""
t3_time_aware_ridge.py — time-aware linear baselines for EWForecast.

In v1.0 the forecasting context is the last k observations (index-based)
and the elapsed time is not a model input. This script checks whether the
time stamps carry signal that a linear head can exploit, under the same
temporal-cutoff protocol as the other EWForecast baselines (cutoff MJD
59215, pairs with >= 5 context points, RidgeCV fitted on pre-cutoff pairs).

  H3  : ridge on [ew_{t-5..t-1}]                                (control)
  T1  : ridge on [ew_{t-5..t-1}, dt_target]                     (+ gap to target)
  T2  : ridge on [ew_{t-5..t-1}, dt_target, 4 in-context gaps]  (fully time-aware)
  T3  : T2 + log1p(dt) versions of the 5 gaps                   (scale-robust)

A fixed-duration context is also probed:
  D1  : ridge on summary statistics of the last 365 days
        [last EW, mean, std, min, max, n_obs, slope, dt_target]
        (pairs with >= 2 observations in the window; persistence is
        recomputed on the same pairs)

Output: <BESS_DOWNSTREAM_DIR>/t3_time_aware_ridge.json
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
K = 5
WINDOW_D = 365.0
ALPHAS = np.logspace(-3, 5, 20)
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
        if len(mjds) < 2:
            continue
        order = np.argsort(mjds)
        series[star] = {"mjds": np.asarray(mjds)[order],
                        "ews": np.asarray(ews, np.float32)[order]}
    return series


def build_pairs(series, side):
    """Pairs with >= K past observations; returns feature dict per pair."""
    rows = []
    for star, d in series.items():
        mjds, ews = d["mjds"], d["ews"]
        positions = (np.where(mjds > CUTOFF_MJD)[0] if side == "test"
                     else np.where(mjds <= CUTOFF_MJD)[0])
        for t in positions:
            if t < K:
                continue
            ctx_ew = ews[t - K:t]
            ctx_mjd = mjds[t - K:t]
            if not (np.all(np.isfinite(ctx_ew)) and np.isfinite(ews[t])):
                continue
            gaps = np.diff(ctx_mjd)                      # 4 in-context gaps
            dt_target = mjds[t] - ctx_mjd[-1]            # gap to target
            # fixed-duration window features (last 365 d before target)
            wmask = (mjds[:t] >= mjds[t] - WINDOW_D)
            wew = ews[:t][wmask]
            wmjd = mjds[:t][wmask]
            if len(wew) >= 2 and (wmjd.max() - wmjd.min()) > 0.5:
                try:
                    slope = float(np.polyfit(wmjd - wmjd[-1],
                                             wew.astype(np.float64), 1)[0])
                except np.linalg.LinAlgError:
                    slope = 0.0
                dur = [float(wew[-1]), float(wew.mean()), float(wew.std()),
                       float(wew.min()), float(wew.max()), float(len(wew)),
                       float(slope), float(dt_target)]
            else:
                dur = None
            rows.append({"y": float(ews[t]), "persist": float(ews[t - 1]),
                         "ctx": ctx_ew.astype(np.float64), "gaps": gaps,
                         "dt": float(dt_target), "dur": dur})
    return rows


def ratio_ci(err_m, err_p, rng):
    n = len(err_m)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = err_m[idx].mean(axis=1) / err_p[idx].mean(axis=1)
    return [float(v) for v in np.percentile(boots, [2.5, 97.5])]


def fit_eval(name, Xtr, ytr, ptr_unused, Xte, yte, pte, res, rng):
    model = Pipeline([("sc", StandardScaler()),
                      ("ridge", RidgeCV(alphas=ALPHAS))])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(yte - pred)))
    mp = float(np.mean(np.abs(yte - pte)))
    ci = ratio_ci(np.abs(yte - pred), np.abs(yte - pte), rng)
    res[name] = {"mae": mae, "mae_persist": mp, "ratio": mae / mp,
                 "ci95": ci, "n_test": int(len(yte))}
    print(f"{name:12s} ratio {mae/mp:.4f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]"
          f"  (MAE {mae:.4f} / persist {mp:.4f}, n={len(yte)})")


def main():
    rng = np.random.default_rng(42)
    series = load_series()
    tr = build_pairs(series, "train")
    te = build_pairs(series, "test")
    print(f"pairs: train {len(tr)}, test {len(te)} (expected 14475 / 7313)")

    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    ptr = np.array([r["persist"] for r in tr])
    pte = np.array([r["persist"] for r in te])

    def X(rows, mode):
        out = []
        for r in rows:
            if mode == "H3":
                out.append(r["ctx"])
            elif mode == "T1":
                out.append(np.concatenate([r["ctx"], [r["dt"]]]))
            elif mode == "T2":
                out.append(np.concatenate([r["ctx"], [r["dt"]], r["gaps"]]))
            elif mode == "T3":
                lg = np.log1p(np.concatenate([[r["dt"]], r["gaps"]]))
                out.append(np.concatenate([r["ctx"], [r["dt"]], r["gaps"], lg]))
        return np.asarray(out)

    res = {}
    for mode in ("H3", "T1", "T2", "T3"):
        fit_eval(mode, X(tr, mode), ytr, ptr, X(te, mode), yte, pte, res, rng)

    # D1 fixed-duration (restrict to pairs where the 365-d window has >=2 obs)
    ktr = [i for i, r in enumerate(tr) if r["dur"] is not None]
    kte = [i for i, r in enumerate(te) if r["dur"] is not None]
    fit_eval("D1_365d",
             np.asarray([tr[i]["dur"] for i in ktr]), ytr[ktr], None,
             np.asarray([te[i]["dur"] for i in kte]), yte[kte], pte[kte],
             res, rng)

    with open(f"{OUT}/t3_time_aware_ridge.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/t3_time_aware_ridge.json")


if __name__ == "__main__":
    main()
