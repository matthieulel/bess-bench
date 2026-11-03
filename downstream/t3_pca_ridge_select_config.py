"""
EWForecast: is the quoted PCA+Ridge configuration what a validation-based
selection would pick?

The paper reports the full grid {PCA(10), PCA(50)} x {context 5, context 10}
on the test period (epochs after MJD 59215) and quotes PCA(10), context 5.
To check that this choice does not rely on the test period, the same grid is
fitted and compared INSIDE the pre-cutoff period: fit up to an earlier date,
validate on the epochs between that date and the official cutoff. Three
earlier dates are tried (2017-01-01, 2018-01-01, 2019-01-01).

The first block re-runs the four configurations with the official cutoff and
must reproduce the MAE ratios of ``t3_pca_ridge_temporal_baseline.json``.

Output: <BESS_DOWNSTREAM_DIR>/t3_pca_ridge_select_config.json
Run   : python downstream/t3_pca_ridge_select_config.py     (CPU, ~1 min)
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Repository-relative defaults; override with the same environment variables
# as the other scripts (BESS_EMB_PATH, BESS_FEATURES_PATH, BESS_DOWNSTREAM_DIR).
_BENCH_ROOT = Path(__file__).resolve().parents[1]
EMB = os.environ.get("BESS_EMB_PATH", str(
    _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"))
FEAT = os.environ.get("BESS_FEATURES_PATH", str(
    _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"))
OUT = os.environ.get("BESS_DOWNSTREAM_DIR", str(
    _BENCH_ROOT / "reference_results" / "downstream"))

CUTOFF_MJD = 59215.0                       # 2021-01-01, official train/test cutoff
VALIDATION_CUTS = {"2017-01-01": 57754.0, "2018-01-01": 58119.0, "2019-01-01": 58484.0}
GRID = [(10, 5), (10, 10), (50, 5), (50, 10)]          # (PCA dimension, context length)
ALPHAS = np.logspace(-3, 5, 20)


def load_aligned():
    """Per star: (mjd, normalised flux, EW) sorted in time, on the scored epochs."""
    emb = torch.load(EMB, map_location="cpu", weights_only=False)
    feat = torch.load(FEAT, map_location="cpu", weights_only=False)
    aligned = {}
    for star in sorted(set(emb) & set(feat)):
        em, fm = emb[star]["mjds"].numpy(), feat[star]["mjds"].numpy()
        flux, ew = feat[star]["flux_norm"].numpy(), feat[star]["ew"].numpy()
        ie, jf = [], []
        for i, m in enumerate(em):
            d = np.abs(fm - m)
            b = int(np.argmin(d))
            if d[b] < 0.01:
                ie.append(i)
                jf.append(b)
        if len(ie) < 3:
            continue
        mjd, f, e = em[ie], flux[jf], ew[jf]
        valid = np.isfinite(e)
        if valid.sum() < 3:
            continue
        order = np.argsort(mjd[valid])
        aligned[star] = (mjd[valid][order], f[valid][order], e[valid][order])
    return aligned


def run(aligned, fit_until, score_until, n_pca, ctx):
    """Fit PCA and the ridge head on epochs <= fit_until, score on the epochs
    in (fit_until, score_until]. Returns (MAE ratio to persistence, n_fit, n_scored)."""
    fit_flux = np.concatenate([f[m <= fit_until] for m, f, _ in aligned.values()
                               if (m <= fit_until).any()])
    pca = PCA(n_components=n_pca, random_state=42).fit(fit_flux)
    x_fit, y_fit, x_sc, y_sc, last = [], [], [], [], []
    for m, f, e in aligned.values():
        keep = m <= score_until
        m, f, e = m[keep], f[keep], e[keep]
        if len(m) == 0:
            continue
        z = pca.transform(f)
        n_fit = int((m <= fit_until).sum())
        for i in range(ctx, n_fit):
            x_fit.append(z[i - ctx:i].ravel())
            y_fit.append(e[i])
        for i in range(max(n_fit, ctx), len(m)):
            x_sc.append(z[i - ctx:i].ravel())
            y_sc.append(e[i])
            last.append(e[i - 1])
    model = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=ALPHAS))])
    model.fit(np.array(x_fit), np.array(y_fit))
    pred = model.predict(np.array(x_sc))
    y_sc, last = np.array(y_sc), np.array(last)
    ratio = float(np.abs(y_sc - pred).mean() / np.abs(y_sc - last).mean())
    return ratio, len(y_fit), len(y_sc)


def main():
    aligned = load_aligned()
    out = {"cutoff_mjd": CUTOFF_MJD, "grid": [f"pca{d}_ctx{c}" for d, c in GRID],
           "n_stars": len(aligned), "test": {}, "validation": {}}

    print("Test period (official cutoff): must reproduce the paper table")
    for d, c in GRID:
        ratio, n_fit, n_sc = run(aligned, CUTOFF_MJD, np.inf, d, c)
        out["test"][f"pca{d}_ctx{c}"] = {"ratio_mae": ratio, "n_train": n_fit, "n_test": n_sc}
        print(f"   PCA({d}) ctx={c}: ratio = {ratio:.4f}   (train {n_fit}, test {n_sc})")

    for label, cut in VALIDATION_CUTS.items():
        print(f"Validation inside the pre-cutoff period: fit <= {label}, "
              f"validate on ({label}, 2021-01-01]")
        res = {}
        for d, c in GRID:
            ratio, n_fit, n_val = run(aligned, cut, CUTOFF_MJD, d, c)
            res[f"pca{d}_ctx{c}"] = {"ratio_mae": ratio, "n_train": n_fit, "n_validation": n_val}
            print(f"   PCA({d}) ctx={c}: ratio = {ratio:.4f}   (train {n_fit}, validation {n_val})")
        best = min(res, key=lambda k: res[k]["ratio_mae"])
        out["validation"][label] = {"fit_until_mjd": cut, "configs": res, "selected": best}
        print(f"   -> selected: {best}")

    path = Path(OUT) / "t3_pca_ridge_select_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
