"""
compute_sigma_ew_and_scree.py — EW measurement floor and PCA variance decay.

1) Per-spectrum equivalent-width uncertainty (Vollmann & Eversberg 2006):

       sigma(EW) = sqrt(1 + Fc_bar / F_bar) * (dlambda - EW) / SNR

   with Fc_bar = 1 (normalised continuum), F_bar = mean normalised flux
   over the +-30 A integration window, dlambda = 60 A and SNR = the
   per-spectrum DER_SNR shipped with the release. Reported over three
   populations:
     * all aligned scored spectra,
     * all epochs of the stars with >= 30 scored epochs,
     * the EWForecast test targets (post-cutoff epochs with >= 5 past
       observations, i.e. the 7,313 pairs of PCA(10) ctx=5).

2) Cumulative variance explained by the PCA basis of the normalised flux
   (the figure itself is drawn by paper/figure_scripts/fig_pca_scree.py).

Output: <BESS_DOWNSTREAM_DIR>/sigma_ew_and_scree.json
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

_BENCH_ROOT = Path(__file__).resolve().parents[1]
EMB = os.environ.get("BESS_EMB_PATH", str(
    _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"))
FEAT = os.environ.get("BESS_FEATURES_PATH", str(
    _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"))
OUT = os.environ.get("BESS_DOWNSTREAM_DIR", str(
    _BENCH_ROOT / "reference_results" / "downstream"))

EW_WINDOW_A = 60.0        # EW integration window: +-30 A
N_BINS = 128
CUTOFF_MJD = 59215.0
CTX = 5
MAE_PERSIST = 1.4723      # A, persistence on the 7,313 ctx=5 test pairs
MAE_BEST = 1.3867         # A, PCA(10)+ridge ctx=5 on the same pairs


def stats(x):
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(q[1]),
            "iqr": [float(q[0]), float(q[2])],
            "mean": float(np.mean(x)), "p95": float(np.percentile(x, 95))}


def main():
    emb = torch.load(EMB, map_location="cpu", weights_only=False)
    feat = torch.load(FEAT, map_location="cpu", weights_only=False)

    lam = np.linspace(-50, 50, N_BINS)
    ew_mask = np.abs(lam) <= 30.0

    sig, ews, snrs_all, stars, flux_all = [], [], [], [], []
    sig_test, ew_test, stars_test = [], [], set()
    for star in sorted(set(emb) & set(feat)):
        es, fs = emb[star], feat[star]
        emjds, fmjds = es["mjds"].numpy(), fs["mjds"].numpy()
        sn = es["snrs"].numpy()
        fx_all, ew_all = fs["flux_norm"].numpy(), fs["ew"].numpy()
        rows = []                       # aligned epochs of this star
        for i, m in enumerate(emjds):
            d = np.abs(fmjds - m)
            b = int(np.argmin(d))
            if d[b] >= 0.01:
                continue
            ew, s = float(ew_all[b]), float(sn[i])
            if not np.isfinite(ew):
                continue
            fbar = float(np.mean(fx_all[b][ew_mask]))
            sigma = (np.sqrt(1.0 + 1.0 / fbar) * (EW_WINDOW_A - ew) / s
                     if (s > 0 and fbar > 0) else np.nan)
            rows.append((float(m), ew, sigma))
            if np.isfinite(sigma):
                sig.append(sigma); ews.append(ew); snrs_all.append(s)
                stars.append(star); flux_all.append(fx_all[b])
        # EWForecast test targets: same star filter and pair construction as
        # t3_pca_ridge_temporal_baseline.py (>= 3 epochs, ctx past points).
        if len(rows) >= 3:
            rows.sort(key=lambda r: r[0])
            for t, (m, ew, sigma) in enumerate(rows):
                if m > CUTOFF_MJD and t >= CTX and np.isfinite(sigma):
                    sig_test.append(sigma); ew_test.append(abs(ew))
                    stars_test.add(star)

    sig, ews, stars = np.asarray(sig), np.asarray(ews), np.asarray(stars)
    flux_all = np.asarray(flux_all, dtype=np.float32)
    sig_test = np.asarray(sig_test)

    res = {"sigma_ew_A_all": stats(sig),
           "abs_ew_A_all": stats(np.abs(ews)),
           "snr_all": stats(np.asarray(snrs_all)),
           "n_stars_all": int(len(set(stars.tolist())))}

    cnt = Counter(stars.tolist())
    ge30 = np.isin(stars, sorted(s for s, c in cnt.items() if c >= 30))
    res["n_stars_ge30"] = int(sum(1 for c in cnt.values() if c >= 30))
    res["sigma_ew_A_ge30"] = stats(sig[ge30])
    res["abs_ew_A_ge30"] = stats(np.abs(ews[ge30]))

    res["n_stars_forecast_test"] = len(stars_test)
    res["sigma_ew_A_forecast_test_targets"] = stats(sig_test)
    res["abs_ew_A_forecast_test_targets"] = stats(np.asarray(ew_test))

    red = MAE_PERSIST - MAE_BEST
    res["mae_persist_A"] = MAE_PERSIST
    res["mae_best_A"] = MAE_BEST
    res["mae_reduction_A"] = round(red, 4)
    for key in ("all", "ge30", "forecast_test_targets"):
        med = res[f"sigma_ew_A_{key}"]["median"]
        res[f"floor_summary_{key}"] = {
            "persistence_over_floor": round(MAE_PERSIST / med, 2),
            "reducible_error_A": round(MAE_PERSIST - med, 3),
            "fraction_of_headroom_closed": round(red / (MAE_PERSIST - med), 3),
            "reduction_over_floor": round(red / med, 2)}

    pca = PCA(n_components=64, random_state=0).fit(flux_all)
    cum = np.cumsum(pca.explained_variance_ratio_)
    res["pca"] = {"evr_first10": [float(v) for v in pca.explained_variance_ratio_[:10]],
                  "cum_var_pca10": float(cum[9]),
                  "cum_var_pca32": float(cum[31]),
                  "cum_var_pca50": float(cum[49]),
                  "n_comp_for_99pct": int(np.searchsorted(cum, 0.99) + 1)}

    print(json.dumps(res, indent=2))
    Path(OUT).mkdir(parents=True, exist_ok=True)
    with open(f"{OUT}/sigma_ew_and_scree.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/sigma_ew_and_scree.json")


if __name__ == "__main__":
    main()
