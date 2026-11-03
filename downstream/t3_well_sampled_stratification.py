"""
EWForecast: results on the well-sampled stars only.

EWForecast is scored on every star that offers a context window (621 stars
for a context of 5), not only on the stars with a long record. This script
splits the per-pair predictions already stored by the baselines into

  * the stars with at least 30 scored H-alpha epochs (193 stars), and
  * the other stars,

and recomputes the MAE ratio to persistence and its bootstrap CI95 on each
group. No model is re-run.

Inputs : <BESS_DOWNSTREAM_DIR>/t3_pca_ridge_temporal_baseline.json
         <BESS_DOWNSTREAM_DIR>/t3_z_ridge_temporal_baseline_ctx{5,10}.json
         <BESS_DOWNSTREAM_DIR>/t3_ts_fm_baselines.json   (optional)
Output : <BESS_DOWNSTREAM_DIR>/t3_well_sampled_stratification.json
Run    : python downstream/t3_well_sampled_stratification.py     (CPU, < 1 min)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

# Repository-relative defaults; override with the same environment variables
# as the other scripts (BESS_EMB_PATH, BESS_FEATURES_PATH, BESS_DOWNSTREAM_DIR).
_BENCH_ROOT = Path(__file__).resolve().parents[1]
EMB = os.environ.get("BESS_EMB_PATH", str(
    _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"))
FEAT = os.environ.get("BESS_FEATURES_PATH", str(
    _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"))
OUT = os.environ.get("BESS_DOWNSTREAM_DIR", str(
    _BENCH_ROOT / "reference_results" / "downstream"))

MIN_EPOCHS = 30
N_BOOT = 10_000
BOOT_SEED = 42


def scored_epochs_per_star():
    """Number of scored epochs per star (embedding aligned with a finite EW)."""
    emb = torch.load(EMB, map_location="cpu", weights_only=False)
    feat = torch.load(FEAT, map_location="cpu", weights_only=False)
    counts = {}
    for star in sorted(set(emb) & set(feat)):
        em, fm = emb[star]["mjds"].numpy(), feat[star]["mjds"].numpy()
        ew = feat[star]["ew"].numpy()
        n = 0
        for m in em:
            d = np.abs(fm - m)
            b = int(np.argmin(d))
            if d[b] < 0.01 and np.isfinite(ew[b]):
                n += 1
        counts[star] = n
    return counts


def ratio_ci(pred, mask, rng):
    """MAE ratio to persistence on the selected pairs, with bootstrap CI95."""
    y = np.asarray(pred["y_true"])[mask]
    err_model = np.abs(y - np.asarray(pred["y_pred_model"])[mask])
    err_persist = np.abs(y - np.asarray(pred["y_pred_persist"])[mask])
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    boot = err_model[idx].mean(axis=1) / err_persist[idx].mean(axis=1)
    return {
        "n_pairs": int(len(y)),
        "mae_model": float(err_model.mean()),
        "mae_persist": float(err_persist.mean()),
        "ratio_mae": float(err_model.mean() / err_persist.mean()),
        "ratio_mae_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main():
    counts = scored_epochs_per_star()
    well = {s for s, n in counts.items() if n >= MIN_EPOCHS}
    print(f"{len(well)} stars with >= {MIN_EPOCHS} scored epochs "
          f"(of {sum(n > 0 for n in counts.values())} scored stars)")

    sources = {}
    pca = json.loads((Path(OUT) / "t3_pca_ridge_temporal_baseline.json").read_text())
    for key in ("pca10_ctx5", "pca10_ctx10", "pca50_ctx5", "pca50_ctx10"):
        sources[key] = pca[key]["predictions"]
    for ctx in (5, 10):
        z_path = Path(OUT) / f"t3_z_ridge_temporal_baseline_ctx{ctx}.json"
        if z_path.exists():
            per_seed = json.loads(z_path.read_text())["per_seed"]
            sources[f"z128_ctx{ctx}"] = next(iter(per_seed.values()))["predictions"]
    fm_path = Path(OUT) / "t3_ts_fm_baselines.json"
    if fm_path.exists():
        for key, value in json.loads(fm_path.read_text()).items():
            sources[key] = value["predictions"]
    else:
        print(f"  {fm_path.name} not found: foundation models skipped")

    rng = np.random.default_rng(BOOT_SEED)
    out = {"min_epochs": MIN_EPOCHS, "n_well_sampled_stars": len(well),
           "n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED, "baselines": {}}
    print(f"{'baseline':20s} | {'all stars':>24s} | {'well-sampled stars':>24s} | {'other stars':>24s}")
    for key, pred in sources.items():
        stars = np.asarray(pred["star_names"])
        is_well = np.isin(stars, sorted(well))
        res = {
            "n_stars_all": int(len(set(stars.tolist()))),
            "n_stars_well_sampled": int(len(set(stars[is_well].tolist()))),
            "all": ratio_ci(pred, np.ones(len(stars), dtype=bool), rng),
            "well_sampled": ratio_ci(pred, is_well, rng),
            "others": ratio_ci(pred, ~is_well, rng),
        }
        out["baselines"][key] = res
        cells = [f"{res[g]['ratio_mae']:.3f} [{res[g]['ratio_mae_ci95'][0]:.3f}, "
                 f"{res[g]['ratio_mae_ci95'][1]:.3f}]" for g in ("all", "well_sampled", "others")]
        print(f"{key:20s} | {cells[0]:>24s} | {cells[1]:>24s} | {cells[2]:>24s}")

    path = Path(OUT) / "t3_well_sampled_stratification.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
