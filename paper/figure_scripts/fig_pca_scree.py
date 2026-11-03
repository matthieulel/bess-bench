"""Appendix figure — variance decay of the PCA basis on the scored H-alpha slice.

Ten components already explain 99.40% of the normalised-flux variance, yet
every linear readout fails on the shape features (SpecProbe): those
features are non-linear functionals of the flux rather than information
discarded by dimensionality reduction.

Input : reference_results/features/spectral_features.pt  (or $BESS_FEATURES_PATH)
Output: figures/fig_pca_scree.pdf (and .png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (no Type 3), as in _common.setup_style
import numpy as np
import torch
from sklearn.decomposition import PCA

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIG_DIR, ROOT  # noqa: E402

FEATURES = Path(os.environ.get(
    "BESS_FEATURES_PATH",
    str(ROOT / "reference_results" / "features" / "spectral_features.pt")))
OUT = FIG_DIR / "fig_pca_scree"
N_COMP = 64
MARKS = [10, 32, 50]


def load_flux():
    feat = torch.load(FEATURES, map_location="cpu", weights_only=False)
    flux = [fs["flux_norm"].numpy() for fs in feat.values()]
    return np.concatenate(flux, axis=0).astype(np.float32)


def main():
    flux = load_flux()
    print(f"flux matrix: {flux.shape}")
    pca = PCA(n_components=N_COMP, random_state=42).fit(flux)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))

    ax1.bar(np.arange(1, N_COMP + 1), 100 * ev, color="#4C72B0", width=0.9)
    ax1.set_yscale("log")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%, log)")
    ax1.set_title("Per-component variance", fontsize=10)
    ax1.set_xlim(0.3, N_COMP + 0.7)

    ax2.plot(np.arange(1, N_COMP + 1), 100 * cum, color="#4C72B0", lw=1.6)
    for m, col in zip(MARKS, ["#C44E52", "#55A868", "#8172B2"]):
        ax2.axvline(m, color=col, ls="--", lw=0.9)
        ax2.annotate(f"PCA({m}): {100 * cum[m - 1]:.2f}%",
                     xy=(m, 100 * cum[m - 1]), xytext=(m + 1.5, 99.0 - 0.35 * MARKS.index(m)),
                     color=col, fontsize=8)
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(97.5, 100.05)
    ax2.set_title("Cumulative variance", fontsize=10)

    fig.suptitle("PCA variance decay on the scored H$\\alpha$ slice "
                 "(linear compressibility $\\neq$ physical informativeness)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf")
    fig.savefig(f"{OUT}.png", dpi=200)
    print(f"saved {OUT}.pdf / .png")
    for m in MARKS:
        print(f"  PCA({m:2d}) cumulative explained variance = {100 * cum[m - 1]:.2f}%")


if __name__ == "__main__":
    main()
