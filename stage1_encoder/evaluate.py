"""
evaluate.py — diagnostics of the BeMAE latent space.

Computed on the embeddings of the encoder's training stars and of its
held-out stars (``TrainConfig.test_stars``, eight intensively observed
stars that are never used for pre-training):

  r2_cv_ew          per-star cross-validated R² of EW from the embedding
  knn_intra_star    within-star emission/absorption k-NN accuracy
  cos_centroids     cosine similarity of emission/absorption centroids
  recall_at_10      star retrieval: same-star fraction among the 10
                    nearest neighbours ("_test" = held-out stars only)

Usage:
    cd stage1_encoder
    python evaluate.py --checkpoint runs/Halpha_all_seed42/best.pt
Writes ``eval_results.json`` and ``eval_details.json`` next to the
checkpoint (or to ``--output_dir``).
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ModelConfig, TrainConfig
from dataset import prepare_data
from model import Stage1HalphaModel


@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=50000):
    """
    Extract z_halpha + metadata from an Hα DataLoader.

    Returns:
      embeddings : [N, d_model]  — z_halpha
      stars      : [N]           — star names
      ews        : [N]           — EW computed in the dataset
      snrs       : [N]           — SNR
      mjds       : [N]           — dates MJD
    """
    model.eval()

    all_z = []
    all_star = []
    all_ew = []
    all_snr = []
    all_mjd = []

    n = 0
    for batch in loader:
        if n >= max_samples:
            break

        flux = batch["flux"].to(device)
        wavelengths = batch["wavelengths"].to(device)
        validity = batch["validity"].to(device)

        z = model.get_embeddings(flux, wavelengths, validity)

        all_z.append(z.cpu().numpy())
        all_star.extend(batch["star_name"])
        all_ew.extend([float(x) for x in batch["ew"]])
        all_snr.extend([float(x) for x in batch["snr"]])
        all_mjd.extend([float(x) for x in batch["mjd"]])

        n += flux.shape[0]

    embeddings = np.concatenate(all_z, axis=0)[:max_samples]
    stars = np.array(all_star[:max_samples])
    ews = np.array(all_ew[:max_samples])
    snrs = np.array(all_snr[:max_samples])
    mjds = np.array(all_mjd[:max_samples])

    return embeddings, stars, ews, snrs, mjds


def eval_r2_cv_ew(embeddings, stars, ews, min_obs=50):
    """    Per-star cross-validated R² of EW predicted from the embeddings.

    For each star with at least ``min_obs`` spectra and a non-constant EW:
    StandardScaler -> PCA(10) -> Ridge(alpha=1), 5-fold CV *within the
    star*. Returns the summary over stars and the per-star details.
    """
    # Group by star
    star_indices = defaultdict(list)
    for i, s in enumerate(stars):
        if np.isfinite(ews[i]):
            star_indices[s].append(i)

    results_per_star = []
    all_r2 = []

    for star, indices in sorted(star_indices.items()):
        if len(indices) < min_obs:
            continue

        X = embeddings[indices]
        y = ews[indices]

        # Check variance of y
        if np.std(y) < 0.1:
            continue  # EW constant → R² non informatif

        # Pipeline : scale → PCA 10D → Ridge
        n_components = min(10, X.shape[1], len(indices) - 2)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("ridge", Ridge(alpha=1.0)),
        ])

        # 5-fold CV
        n_folds = min(5, len(indices))
        scores = cross_val_score(pipe, X, y, cv=n_folds, scoring="r2")
        r2_mean = float(np.mean(scores))

        results_per_star.append({
            "star": star,
            "r2_cv": r2_mean,
            "n_obs": len(indices),
            "ew_std": float(np.std(y)),
            "ew_range": [float(np.min(y)), float(np.max(y))],
        })
        all_r2.append(r2_mean)

    all_r2 = np.array(all_r2)

    # Summary
    summary = {
        "n_stars_tested": len(all_r2),
        "r2_mean": float(np.mean(all_r2)) if len(all_r2) > 0 else float("nan"),
        "r2_median": float(np.median(all_r2)) if len(all_r2) > 0 else float("nan"),
        "r2_std": float(np.std(all_r2)) if len(all_r2) > 0 else float("nan"),
        "n_above_0": int((all_r2 > 0).sum()),
        "n_above_0.1": int((all_r2 > 0.1).sum()),
        "n_above_0.3": int((all_r2 > 0.3).sum()),
        "pct_above_0": float((all_r2 > 0).mean() * 100) if len(all_r2) > 0 else 0,
        "pct_above_0.1": float((all_r2 > 0.1).mean() * 100) if len(all_r2) > 0 else 0,
        "top_stars": sorted(results_per_star, key=lambda x: -x["r2_cv"])[:10],
        "worst_stars": sorted(results_per_star, key=lambda x: x["r2_cv"])[:5],
    }

    return summary, results_per_star


def eval_knn_intra_star(embeddings, stars, ews,
                        ew_emission=-3.0, ew_absorption=3.0,
                        k=5, min_per_class=5):
    """    Within-star emission/absorption separation.

    For each star with at least ``min_per_class`` spectra in emission
    (EW < ew_emission) and in absorption (EW > ew_absorption): leave-one-out
    k-nearest-neighbour classification (cosine distance, k=5) among the
    star's own spectra. Reports the accuracy and the gain over the
    majority-class baseline, averaged over stars.

    This is a diagnostic of state separation inside a star; it is not a
    stellar-identity metric (see ``eval_retrieval`` for that).
    """
    star_indices = defaultdict(list)
    for i, s in enumerate(stars):
        if np.isfinite(ews[i]):
            star_indices[s].append(i)

    results = []

    for star, indices in sorted(star_indices.items()):
        indices = np.array(indices)
        star_ews = ews[indices]

        em_mask = star_ews < ew_emission
        ab_mask = star_ews > ew_absorption

        n_em = em_mask.sum()
        n_ab = ab_mask.sum()

        if n_em < min_per_class or n_ab < min_per_class:
            continue

        # Emission + absorption spectra of this star
        sel = em_mask | ab_mask
        X = embeddings[indices[sel]]
        y = em_mask[sel].astype(int)  # 1 = emission, 0 = absorption

        # Baseline : majority class
        baseline = max(y.mean(), 1 - y.mean())

        # KNN LOO
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X)
        _, knn_indices = nn.kneighbors(X)

        correct = 0
        for i in range(len(X)):
            neighbors = knn_indices[i, 1:]  # Exclude self
            pred = int(y[neighbors].mean() >= 0.5)
            if pred == y[i]:
                correct += 1

        accuracy = correct / len(X)

        results.append({
            "star": star,
            "knn_acc": accuracy,
            "baseline": baseline,
            "delta": accuracy - baseline,
            "n_em": int(n_em),
            "n_ab": int(n_ab),
            "n_total": int(sel.sum()),
        })

    # Summary
    if len(results) == 0:
        return {"n_stars_tested": 0}, results

    deltas = np.array([r["delta"] for r in results])
    accs = np.array([r["knn_acc"] for r in results])

    summary = {
        "n_stars_tested": len(results),
        "mean_accuracy": float(np.mean(accs)),
        "mean_delta": float(np.mean(deltas)),
        "n_beat_baseline_5pct": int((deltas > 0.05).sum()),
        "pct_beat_baseline_5pct": float((deltas > 0.05).mean() * 100),
        "n_beat_baseline_10pct": int((deltas > 0.10).sum()),
        "top_stars": sorted(results, key=lambda x: -x["delta"])[:10],
        "worst_stars": sorted(results, key=lambda x: x["delta"])[:5],
    }

    return summary, results


def eval_cos_centroids(embeddings, stars, ews,
                       ew_emission=-3.0, ew_absorption=3.0,
                       min_per_class=5):
    """    Cosine similarity between the emission and absorption centroids of
    each star that has at least ``min_per_class`` spectra in both states.
    """
    star_indices = defaultdict(list)
    for i, s in enumerate(stars):
        if np.isfinite(ews[i]):
            star_indices[s].append(i)

    results = []

    for star, indices in sorted(star_indices.items()):
        indices = np.array(indices)
        star_ews = ews[indices]

        em_mask = star_ews < ew_emission
        ab_mask = star_ews > ew_absorption

        if em_mask.sum() < min_per_class or ab_mask.sum() < min_per_class:
            continue

        z_em = embeddings[indices[em_mask]]
        z_ab = embeddings[indices[ab_mask]]

        centroid_em = z_em.mean(axis=0)
        centroid_ab = z_ab.mean(axis=0)

        # Cosine similarity
        cos = np.dot(centroid_em, centroid_ab) / (
            np.linalg.norm(centroid_em) * np.linalg.norm(centroid_ab) + 1e-10
        )

        # Euclidean distance
        dist = np.linalg.norm(centroid_em - centroid_ab)

        results.append({
            "star": star,
            "cos_sim": float(cos),
            "euclidean_dist": float(dist),
            "n_em": int(em_mask.sum()),
            "n_ab": int(ab_mask.sum()),
        })

    if len(results) == 0:
        return {"n_stars_tested": 0}, results

    cos_vals = np.array([r["cos_sim"] for r in results])

    summary = {
        "n_stars_tested": len(results),
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals)),
        "cos_min": float(np.min(cos_vals)),
        "cos_max": float(np.max(cos_vals)),
        "n_below_0.95": int((cos_vals < 0.95).sum()),
        "n_below_0.90": int((cos_vals < 0.90).sum()),
        "pct_below_0.95": float((cos_vals < 0.95).mean() * 100),
        "top_separated": sorted(results, key=lambda x: x["cos_sim"])[:10],
        "worst_separated": sorted(results, key=lambda x: -x["cos_sim"])[:5],
    }

    return summary, results


def eval_retrieval(embeddings, stars, k=10):
    """    Recall@k for star retrieval: fraction of the k nearest latent
    neighbours (cosine distance) of a spectrum that belong to the same star,
    averaged over all spectra passed in.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    recalls = []
    for i in range(len(embeddings)):
        neighbors = indices[i, 1:]
        same = (stars[neighbors] == stars[i]).sum()
        recalls.append(same / k)

    return float(np.mean(recalls))


def main():
    parser = argparse.ArgumentParser(description="Diagnostics of the BeMAE latent space")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_obs_r2", type=int, default=30,
                        help="Minimum number of spectra per star for the EW R² test")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write eval_results.json (default: checkpoint directory)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json") as f:
        config = json.load(f)
    model_cfg = ModelConfig(**{k: v for k, v in config["model"].items()
                               if k in ModelConfig.__dataclass_fields__})
    train_cfg = TrainConfig(**{k: v for k, v in config["train"].items()
                               if k in TrainConfig.__dataclass_fields__})

    print("Loading the H-alpha slice...")
    train_loader, test_loader = prepare_data(train_cfg)
    # The training loader shuffles and drops the last incomplete batch; the
    # diagnostics must see every spectrum, in a fixed order.
    train_loader = DataLoader(train_loader.dataset, batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=train_cfg.num_workers,
                              drop_last=False)

    model = Stage1HalphaModel(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  checkpoint epoch {ckpt.get('epoch', '?')}, "
          f"validation loss {ckpt.get('val_mae_loss', '?')}")

    emb_train, stars_train, ews_train, _, _ = extract_embeddings(
        model, train_loader, device, max_samples=args.max_samples)
    emb_test, stars_test, ews_test, _, _ = extract_embeddings(
        model, test_loader, device, max_samples=args.max_samples)
    # Self-supervised pre-training never sees EW or state labels, so the
    # diagnostics are reported both on the held-out stars alone and on all
    # stars (train + held-out).
    emb_all = np.concatenate([emb_train, emb_test], axis=0)
    stars_all = np.concatenate([stars_train, stars_test])
    ews_all = np.concatenate([ews_train, ews_test])
    print(f"  embeddings: {len(emb_train):,} (training stars) + {len(emb_test):,} "
          f"(held-out stars: {sorted(set(stars_test.tolist()))})")

    results = {
        "n_spectra_train_stars": int(len(emb_train)),
        "n_spectra_heldout_stars": int(len(emb_test)),
        "n_train_stars": int(len(set(stars_train.tolist()))),
        "n_heldout_stars": int(len(set(stars_test.tolist()))),
        "heldout_stars": sorted(set(stars_test.tolist())),
    }

    r2_test, r2_details_test = eval_r2_cv_ew(emb_test, stars_test, ews_test, min_obs=args.min_obs_r2)
    r2_all, r2_details_all = eval_r2_cv_ew(emb_all, stars_all, ews_all, min_obs=args.min_obs_r2)
    results["r2_cv_ew_test"], results["r2_cv_ew_all"] = r2_test, r2_all

    knn_test, _ = eval_knn_intra_star(emb_test, stars_test, ews_test)
    knn_all, knn_details = eval_knn_intra_star(emb_all, stars_all, ews_all)
    results["knn_intra_star_test"], results["knn_intra_star"] = knn_test, knn_all

    cos_test, _ = eval_cos_centroids(emb_test, stars_test, ews_test)
    cos_all, cos_details = eval_cos_centroids(emb_all, stars_all, ews_all)
    results["cos_centroids_test"], results["cos_centroids"] = cos_test, cos_all

    results["recall_at_10_test"] = eval_retrieval(emb_test, stars_test, k=10)
    results["recall_at_10"] = eval_retrieval(emb_all, stars_all, k=10)

    print(f"  per-star R2(EW)      : held-out {r2_test['r2_mean']:.4f} ({r2_test['n_stars_tested']} stars) | "
          f"all {r2_all['r2_mean']:.4f} ({r2_all['n_stars_tested']} stars)")
    print(f"  within-star kNN acc. : {knn_all.get('mean_accuracy', float('nan')):.4f} "
          f"({knn_all['n_stars_tested']} stars with both states)")
    print(f"  centroid cosine      : {cos_all.get('cos_mean', float('nan')):.4f}")
    print(f"  Recall@10            : held-out stars {results['recall_at_10_test']:.4f} | "
          f"all stars {results['recall_at_10']:.4f}")

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / "eval_details.json", "w") as f:
        json.dump({"r2_per_star_test": r2_details_test, "r2_per_star_all": r2_details_all,
                   "knn_per_star": knn_details, "cos_per_star": cos_details},
                  f, indent=2, default=str)
    print(f"Results saved to {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
