"""t3_timemoe_zeroshot.py — Time-MoE zero-shot on EWForecast (Shi et al. 2025).

Same protocol and population as ``t3_ts_fm_baselines.py`` and the Moirai
runs: temporal cutoff MJD 59215, causal contexts of up to 64 past EW
values, stars with >= 3 finite aligned epochs, 1-step-ahead point forecast,
MAE ratio against persistence on the identical target pairs (n = 8,118,
persistence MAE 1.6307 A), paired bootstrap CI95 (10,000 resamples).

Time-MoE is a decoder-only mixture-of-experts forecaster served through
HuggingFace transformers (``trust_remote_code``). It expects per-series
standardised inputs; contexts are grouped by length so that no padding or
attention mask is involved. Prediction uses a direct forward pass with
``use_cache=False`` (the released remote code relies on a cache API removed
from recent transformers versions); for a 1-step horizon the last-position
output is the forecast (checked on a constant series).

Output: <BESS_DOWNSTREAM_DIR>/t3_timemoe_zeroshot.json
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

CUTOFF_MJD = 59215.0
MAX_CONTEXT = 64
N_BOOT = 10_000
BATCH = 256
MODELS = {"50M": "Maple728/TimeMoE-50M", "200M": "Maple728/TimeMoE-200M"}


def load_series():
    emb = torch.load(EMB, map_location="cpu", weights_only=False)
    feat = torch.load(FEAT, map_location="cpu", weights_only=False)
    series = {}
    for star in sorted(set(emb) & set(feat)):
        es, fs = emb[star], feat[star]
        emjds, fmjds = es["mjds"].numpy(), fs["mjds"].numpy()
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
        o = np.argsort(mjds)
        series[star] = {"mjds": np.asarray(mjds)[o],
                        "ews": np.asarray(ews, np.float32)[o]}
    return series


def build_test_pairs(series):
    contexts, y, persist = [], [], []
    for d in series.values():
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


def timemoe_predict(hf_id, contexts):
    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {hf_id} on {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map=device, trust_remote_code=True).eval()

    preds = np.empty(len(contexts), dtype=np.float32)
    by_len = {}
    for i, c in enumerate(contexts):
        by_len.setdefault(len(c), []).append(i)

    done = 0
    for L, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), BATCH):
            chunk = idxs[s:s + BATCH]
            x = torch.tensor(np.stack([contexts[i] for i in chunk]),
                             dtype=torch.float32, device=device)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            std = torch.where(std > 1e-6, std, torch.ones_like(std))
            with torch.no_grad():
                # Direct forward pass instead of generate(): the released
                # Time-MoE remote code calls DynamicCache.get_usable_length(),
                # removed from recent transformers versions. For a 1-step
                # horizon the last-position output is the forecast (a
                # constant series at 5.0 predicts exactly 5.0).
                out = model(input_ids=(x - mean) / std, use_cache=False,
                            return_dict=True)
            p = (out.logits[:, -1, 0:1] * std + mean).squeeze(-1).float().cpu().numpy()
            for j, i in enumerate(chunk):
                preds[i] = p[j]
            done += len(chunk)
        if done % 2000 < BATCH:
            print(f"  {done}/{len(contexts)}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds


def ratio_ci(err_m, err_p, rng):
    n = len(err_m)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return [float(v) for v in np.percentile(
        err_m[idx].mean(axis=1) / err_p[idx].mean(axis=1), [2.5, 97.5])]


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    contexts, y, persist = build_test_pairs(load_series())
    mae_p = float(np.mean(np.abs(y - persist)))
    print(f"pairs: {len(y)} (expected 8118) | persistence {mae_p:.4f} "
          f"(expected 1.6307)", flush=True)

    res = {"n_test": int(len(y)), "mae_persist": mae_p,
           "protocol": "cutoff MJD 59215, context <=64, 1-step, published "
                       "star filter (>=3 epochs), seeds 42",
           "published_ts_fm_same_pairs": {"chronos_bolt_base": 0.9676,
                                          "timesfm_2_0_500m": 0.9689}}
    for name, hf_id in MODELS.items():
        preds = timemoe_predict(hf_id, contexts)
        if not np.all(np.isfinite(preds)):
            print(f"  !! {np.sum(~np.isfinite(preds))} non-finite predictions")
        mae = float(np.mean(np.abs(y - preds)))
        res[f"time_moe_{name}"] = {
            "mae": mae, "ratio": mae / mae_p,
            "ci95": ratio_ci(np.abs(y - preds), np.abs(y - persist), rng)}
        print(f"Time-MoE-{name}: MAE {mae:.4f}  ratio {mae/mae_p:.4f}  "
              f"CI95 {res[f'time_moe_{name}']['ci95']}", flush=True)

    with open(f"{OUT}/t3_timemoe_zeroshot.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"saved {OUT}/t3_timemoe_zeroshot.json")


if __name__ == "__main__":
    main()
