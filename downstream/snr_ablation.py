"""
snr_ablation.py — SpecProbe per-SNR-bin diagnostic (paper appendix).

For each spectral feature and each frozen representation
(z = BeMAE-128, PCA(10), raw flux 128), runs the SpecProbe ridge probe
with out-of-fold predictions on the full scored slice, then reports
R² per SNR bin (low / medium / good / excellent) and the pooled
out-of-fold R² ("global").

The protocol is the SpecProbe one: 5-fold ``GroupKFold`` on ``star_id``
(default) and PCA(10) fitted on the normalised flux exactly as in
``probe_features.py``. ``--ungrouped`` restores the spectrum-level
``KFold(shuffle=True)`` of the original submission.

Output: <output>.json + a console summary.

Usage:
    python downstream/snr_ablation.py \
        --embeddings reference_results/embeddings/star_data.pt \
        --features   reference_results/features/spectral_features.pt \
        --seed 42 --output reference_results/downstream/snr_ablation_seed42.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["fwhm", "central_depth", "delta_v", "vr_ratio"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SNR_BINS = [(0, 60, "low"), (60, 120, "medium"),
            (120, 300, "good"), (300, np.inf, "excellent")]


def align(emb_data, feat_data, mjd_tol=0.01):
    X, flux, snrs, groups, feats = [], [], [], [], {n: [] for n in FEATURE_NAMES}
    for star in sorted(set(emb_data) & set(feat_data)):
        ed, fd = emb_data[star], feat_data[star]
        em = ed["mjds"].numpy(); fm = fd["mjds"].numpy()
        ev = ed["embeddings"].numpy(); sn = ed["snrs"].numpy()
        flx = fd["flux_norm"].numpy()
        for i, m in enumerate(em):
            j = int(np.argmin(np.abs(fm - m)))
            if abs(fm[j] - m) < mjd_tol:
                X.append(ev[i]); flux.append(flx[j]); snrs.append(sn[i])
                groups.append(star)
                for n in FEATURE_NAMES:
                    feats[n].append(float(fd[n][j].item()))
    return (np.asarray(X, np.float32),
            np.asarray(flux, np.float32),
            np.asarray(snrs, np.float32),
            np.asarray(groups),
            {n: np.asarray(v, np.float32) for n, v in feats.items()})


def probe_oof(X, y, groups, seed=42, n_folds=5, grouped=True):
    """Out-of-fold ridge predictions under the SpecProbe CV protocol."""
    valid = np.isfinite(y)
    Xv, yv = X[valid], y[valid]
    pipe = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=ALPHAS))])
    if grouped:
        yhat = cross_val_predict(pipe, Xv, yv, cv=GroupKFold(n_splits=n_folds),
                                 groups=groups[valid])
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yhat = cross_val_predict(pipe, Xv, yv, cv=cv)
    return valid, yhat, yv


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--features",
                   default=str(Path(__file__).resolve().parents[1]
                               / "reference_results" / "features"
                               / "spectral_features.pt"))
    p.add_argument("--seed", type=int, default=42,
                   help="Encoder seed label written to the output; with "
                        "--ungrouped it also seeds the KFold shuffle.")
    p.add_argument("--ungrouped", action="store_true",
                   help="Spectrum-level KFold(shuffle=True) instead of the "
                        "default GroupKFold on star_id (original submission).")
    p.add_argument("--output", default="snr_ablation.json")
    args = p.parse_args()
    grouped = not args.ungrouped

    print(f"[snr_ablation] seed={args.seed}")
    emb = torch.load(args.embeddings, weights_only=False)
    feat = torch.load(args.features, weights_only=False)
    X, flux, snrs, groups, feats = align(emb, feat)
    print(f"  aligned {len(X):,} spectra from {len(set(groups.tolist())):,} stars")
    print(f"  CV: {'GroupKFold(star_id)' if grouped else 'KFold(shuffle=True)'}")
    print(f"  SNR median={np.median(snrs):.0f} q25={np.quantile(snrs,0.25):.0f} q75={np.quantile(snrs,0.75):.0f}")

    # Same PCA(10) as SpecProbe (probe_features.py): fitted on the
    # normalised flux, no prior standardisation.
    pca10 = PCA(n_components=10, random_state=0).fit_transform(flux)

    representations = {"BeMAE": X, "PCA10": pca10, "Flux128": flux}
    out = {"task": "snr_ablation", "seed": args.seed,
           "cv": "GroupKFold(star_id)" if grouped else "KFold(shuffle=True)",
           "n_spectra": int(len(X)),
           "n_stars": int(len(set(groups.tolist()))),
           "bins": {}, "global": {}}

    # bin counts
    for lo, hi, name in SNR_BINS:
        mask = (snrs >= lo) & (snrs < hi)
        out["bins"][name] = {"snr_lo": float(lo),
                             "snr_hi": (None if not np.isfinite(hi) else float(hi)),
                             "n": int(mask.sum())}
        print(f"  bin {name:9s} [{lo:>4.0f}, {hi:>5.0f}): n={mask.sum():,}")

    # for each rep × feature: get OOF predictions and stratify R² by bin
    for rep_name, Xrep in representations.items():
        out["bins"][rep_name] = {} if False else None  # placeholder
        for feat_name in FEATURE_NAMES:
            y = feats[feat_name]
            valid, yhat, yv = probe_oof(Xrep, y, groups, seed=args.seed,
                                        grouped=grouped)
            global_r2 = r2(yv, yhat)
            out["global"].setdefault(rep_name, {})[feat_name] = global_r2
            # per-bin R² (use SNR of valid spectra)
            snrs_v = snrs[valid]
            for lo, hi, bname in SNR_BINS:
                bm = (snrs_v >= lo) & (snrs_v < hi)
                if bm.sum() < 20:
                    rr = float("nan")
                else:
                    rr = r2(yv[bm], yhat[bm])
                out["bins"].setdefault(bname, {}) \
                    .setdefault(rep_name, {})[feat_name] = rr

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # console summary table
    print("\n  R² per bin × representation × feature")
    print(f"  {'feat':14s} | {'rep':8s} | " +
          " | ".join(f"{n:>9s}" for _, _, n in SNR_BINS) + " | global")
    for feat_name in FEATURE_NAMES:
        for rep_name in ("BeMAE", "PCA10", "Flux128"):
            row = [out["bins"][bn][rep_name][feat_name]
                   for _, _, bn in SNR_BINS]
            row_s = " | ".join(f"{v:>9.3f}" if v == v else f"{'    n/a':>9s}"
                               for v in row)
            print(f"  {feat_name:14s} | {rep_name:8s} | {row_s} | "
                  f"{out['global'][rep_name][feat_name]:.3f}")
    print(f"\n  saved → {args.output}")


if __name__ == "__main__":
    main()
