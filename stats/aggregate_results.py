#!/usr/bin/env python3
"""
aggregate_results.py — Aggregates result JSONs and produces
table LaTeX for the paper.

Inputs (repository-relative):
  reference_results/stage1_encoder/multiseed_summary.json          (BeMAE diagnostics)
  reference_results/downstream/probe_features_results_seed*.json   (SpecProbe, 3 encoder seeds)
  reference_results/downstream/probe_t2_hbeta_to_halpha.json       (LineTransfer)
  reference_results/downstream/snr_ablation_seed*.json             (SpecProbe per SNR bin)
  reference_results/downstream/t3_pca_ridge_temporal_baseline.json (EWForecast PCA+ridge)
  reference_results/downstream/t3_z_ridge_temporal_baseline_ctx*.json (EWForecast z+ridge)
  reference_results/downstream/t3_ts_fm_baselines.json             (zero-shot Chronos-Bolt, TimesFM)
  reference_results/downstream/t3_{moirai,moiraimoe,timemoe}_zeroshot.json (further zero-shot models)
  reference_results/downstream/t3_chronos_{trained_head,zeroshot_samepairs}.json
  reference_results/downstream/t3_transformer_head.json            (causal Transformer head)
  reference_results/stats/t3_bootstrap_cis.json                    (CI95 of the MAE ratios)

Outputs (paper/tables/, or $BESS_TABLES_DIR):
  table_stage1.tex            table_t1_probes.tex      table_t1_full.tex
  table_t2_linetransfer.tex   table_snr.tex            table_t3_baselines.tex
  table_t3_well_sampled.tex
  summary.json   (every number of the tables, machine-readable)

Optional inputs are skipped when absent, so the same script renders the
seed-42 tables of ``scripts/reproduce_paper_seed42.sh``.
"""
from __future__ import annotations

import json
import os
import statistics as _stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Output dir: defaults to paper/tables/ (canonical committed copy);
# override via BESS_TABLES_DIR for reproduction runs.
OUT_TABLES = Path(os.environ.get(
    "BESS_TABLES_DIR", str(ROOT / "paper" / "tables")))
OUT_TABLES.mkdir(parents=True, exist_ok=True)
# Source for downstream JSONs (probe_features_results_seed*, t3_*).
# Defaults to the committed reference_results/ but can be redirected to
# results_reproduced/downstream/ to render tables from a fresh repro run.
DOWNSTREAM_DIR = Path(os.environ.get(
    "BESS_DOWNSTREAM_DIR", str(ROOT / "reference_results" / "downstream")))


def load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


# ─── Stage 1 encoder (BeMAE) ────────────────────────────────────────────────
def table_stage1() -> tuple[str, dict]:
    """Multi-seed aggregate over seeds {42,123,456}.

    Prefers ``runs/multiseed_summary.json`` (new pipeline). Falls back to the
    legacy single-run ``runs/Halpha_all/eval_results.json`` if present.
    """
    multi = ROOT / "reference_results" / "stage1_encoder" / "multiseed_summary.json"
    if multi.exists():
        d = load_json(multi)
        agg = d["aggregate"]
        def fmt(key):
            a = agg.get(key, {})
            m, s, n = a.get("mean"), a.get("std"), a.get("n", 0)
            if m is None:
                return "n/a", {"mean": None, "std": None, "n": n}
            return f"{m:.3f} $\\pm$ {s:.3f}", {"mean": m, "std": s, "n": n}
        label_keys = [
            ("kNN intra-star acc.",                     "knn_intra_star"),
            ("Recall@10 (held-out)",                    "recall_at_10_test"),
        ]
        metrics = {}
        rows_tex = []
        for label, key in label_keys:
            txt, val = fmt(key)
            metrics[label] = val
            rows_tex.append(f"  {label.ljust(26)} & {txt} \\\\")
        rows = "\n".join(rows_tex)
        tex = (
            "\\begin{tabular}{lr}\n"
            "  \\toprule\n"
            "  Metric & Value (mean $\\pm$ std, 3 seeds) \\\\\n"
            "  \\midrule\n"
            f"{rows}\n"
            "  \\bottomrule\n"
            "\\end{tabular}\n"
        )
        metrics["_seeds"] = d.get("seeds", [])
        metrics["_per_seed"] = d.get("per_seed", [])
        return tex, metrics

    # Fallback: legacy single-seed path
    d = load_json(ROOT / "reference_results" / "stage1_encoder" / "Halpha_all_seed42" / "eval_results.json")
    metrics = {
        "R^2 CV (EW, all stars)":   d["r2_cv_ew_all"]["r2_mean"],
        "R^2 CV median (all)":      d["r2_cv_ew_all"]["r2_median"],
        "kNN intra-star acc.":      d["knn_intra_star"]["mean_accuracy"],
        "Centroid cos. sim. (mean)": d["cos_centroids"]["cos_mean"],
        "Recall@10 (held-out)":     d["recall_at_10_test"],
    }
    rows = "\n".join(
        f"  {k.ljust(26)} & {v:.3f} \\\\"
        for k, v in metrics.items()
    )
    tex = (
        "\\begin{tabular}{lr}\n"
        "  \\toprule\n"
        "  Metric & Test value \\\\\n"
        "  \\midrule\n"
        f"{rows}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    return tex, metrics


# ─── helpers ────────────────────────────────────────────────────────────────
def _tt(name: str) -> str:
    """LaTeX-escape a feature name."""
    return name.replace("_", "\\_")


def _n(x) -> str:
    """Integer with a LaTeX thousands separator (26{,}858)."""
    return f"{int(x):,}".replace(",", "{,}")


def _probe_seed_files() -> list[dict]:
    files = [DOWNSTREAM_DIR / f"probe_features_results_seed{s}.json" for s in (42, 123, 456)]
    files = [p for p in files if p.exists()]
    if files:
        return [load_json(p) for p in files]
    return [load_json(DOWNSTREAM_DIR / "probe_features_results.json")]


T1_REPS = [
    ("z (128D)",           "z",      "BeMAE $z$ (128D)"),
    ("PCA(10)",            "pca10",  "PCA(10)"),
    ("PCA(32)",            "pca32",  "PCA(32)"),
    ("PCA(50)",            "pca50",  "PCA(50)"),
    ("raw-flux-128D",      "flux",   "Raw flux (128D)"),
    ("handcrafted-5D-LOO", "loo5",   "5-D LOO"),
]


def _t1_cell(per_seed: list[dict], feat: str, rep: str) -> dict:
    means = [dd["results"][feat][rep]["r2_mean"] for dd in per_seed]
    m = _stats.mean(means)
    # Sample standard deviation across encoder seeds; a single-seed run
    # falls back to the across-fold standard deviation.
    s = _stats.stdev(means) if len(means) > 1 else per_seed[0]["results"][feat][rep]["r2_std"]
    return {"r2_mean": m, "r2_std": s, "n_seeds": len(means)}


# ─── T1 probes (frozen embeddings vs linear baselines) ──────────────────────
def table_t1_probes() -> tuple[str, dict]:
    """SpecProbe headline table: mean ± std over encoder seeds {42,123,456}."""
    per_seed = _probe_seed_files()
    n_seeds = len(per_seed)
    cv = per_seed[0].get("cv", "KFold(shuffle=True)")
    features = ["fwhm", "central_depth", "delta_v", "vr_ratio"]  # ew / peak_intensity: appendix table
    reps = [r for r in T1_REPS if r[1] != "loo5"]

    summary, lines = {}, []
    for feat in features:
        cells = {short: _t1_cell(per_seed, feat, rep) for rep, short, _ in reps}
        summary[feat] = cells
        best = max(cells, key=lambda k: cells[k]["r2_mean"])
        tex = []
        for _, short, _ in reps:
            c = cells[short]
            t = f"{c['r2_mean']:.3f} $\\pm$ {c['r2_std']:.3f}"
            tex.append(f"\\textbf{{{t}}}" if short == best else t)
        lines.append(f"  {_tt(feat):<20} & " + " & ".join(tex) + " \\\\")

    suffix = (f"% R^2, mean over {n_seeds} encoder seeds +- sample std across seeds; CV = {cv}.\n"
              if n_seeds > 1 else
              f"% R^2 (single encoder seed), mean +- std across CV folds; CV = {cv}.\n")
    tex = (
        "\\begin{tabular}{l" + "c" * len(reps) + "}\n"
        "  \\toprule\n"
        "  Feature & " + " & ".join(lbl for _, _, lbl in reps) + " \\\\\n"
        "  \\midrule\n" + "\n".join(lines) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n" + suffix
    )
    summary["_cv"] = cv
    summary["_n_seeds"] = n_seeds
    return tex, summary


def table_t1_full() -> tuple[str, dict]:
    """Appendix table: all six features x all six representations (means)."""
    per_seed = _probe_seed_files()
    features = ["ew", "peak_intensity", "fwhm", "central_depth", "delta_v", "vr_ratio"]
    summary, lines = {}, []
    for feat in features:
        if feat not in per_seed[0]["results"]:
            continue
        cells = {short: _t1_cell(per_seed, feat, rep) for rep, short, _ in T1_REPS
                 if rep in per_seed[0]["results"][feat]}
        summary[feat] = cells
        ranked = sorted(cells, key=lambda k: -cells[k]["r2_mean"])
        best = ranked[0]
        tie = (len(ranked) > 1 and
               cells[ranked[0]]["r2_mean"] - cells[ranked[1]]["r2_mean"] < 5e-4)
        tex = []
        for _, short, _ in T1_REPS:
            if short not in cells:
                tex.append("--")
                continue
            t = f"{cells[short]['r2_mean']:.3f}"
            tex.append(f"\\textbf{{{t}}}" if (short == best and not tie) else t)
        lines.append(f"  {_tt(feat):<20} & " + " & ".join(tex) + " \\\\")
    tex = (
        "\\begin{tabular}{l" + "c" * len(T1_REPS) + "}\n"
        "  \\toprule\n"
        "  Feature & " + " & ".join(lbl.replace("BeMAE ", "") for _, _, lbl in T1_REPS) + " \\\\\n"
        "  \\midrule\n" + "\n".join(lines) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        f"% Mean R^2 over {len(per_seed)} encoder seed(s); bold = best per feature.\n"
    )
    return tex, summary


# ─── T2 LineTransfer ────────────────────────────────────────────────────────
def table_t2_linetransfer() -> tuple[str, dict]:
    p = DOWNSTREAM_DIR / "probe_t2_hbeta_to_halpha.json"
    if not p.exists():
        return "", {}
    d = load_json(p)
    reps = [("Hβ PCA(10)", "PCA(10)"), ("Hβ PCA(50)", "PCA(50)"), ("Hβ flux (128D)", "Raw flux (128D)")]
    features = ["ew", "peak_intensity", "fwhm", "central_depth", "delta_v", "vr_ratio"]
    summary, lines = {"_n_pairs": d.get("n_aligned_pairs"),
                      "_n_stars": d.get("n_stars_aligned"), "_cv": d.get("cv")}, []
    for feat in features:
        if feat not in d["results"]:
            continue
        r = d["results"][feat]
        cells = {lbl: r[key] for key, lbl in reps if key in r}
        summary[feat] = cells
        n = next(iter(cells.values()))
        tex = [f"{cells[lbl]['r2_mean']:.2f} $\\pm$ {cells[lbl]['r2_std']:.2f}" for _, lbl in reps]
        n_stars = n.get("n_stars")
        lines.append(f"  {_tt(feat):<16} & {_n(n['n_valid'])} & "
                     + (_n(n_stars) if n_stars else "--") + " & " + " & ".join(tex) + " \\\\")
    tex = (
        "\\begin{tabular}{lrrccc}\n"
        "  \\toprule\n"
        "  H$\\alpha$ target & Pairs & Stars & " + " & ".join(lbl for _, lbl in reps) + " \\\\\n"
        "  \\midrule\n" + "\n".join(lines) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        f"% R^2 of a ridge probe on the H-beta window, mean +- std across the 5 folds; CV = {d.get('cv')}.\n"
    )
    return tex, summary


# ─── SpecProbe per SNR bin ──────────────────────────────────────────────────
def table_snr() -> tuple[str, dict]:
    files = [DOWNSTREAM_DIR / f"snr_ablation_seed{s}.json" for s in (42, 123, 456)]
    files = [p for p in files if p.exists()]
    if not files:
        return "", {}
    runs = [load_json(p) for p in files]
    bins = ["low", "medium", "good", "excellent"]
    reps = [("BeMAE", "BeMAE"), ("PCA10", "PCA(10)"), ("Flux128", "Flux128")]
    features = ["fwhm", "central_depth", "delta_v", "vr_ratio"]
    summary = {"_cv": runs[0].get("cv"), "_n_seeds": len(runs),
               "_n": {b: runs[0]["bins"][b]["n"] for b in bins}}
    lines = []
    for feat in features:
        summary[feat] = {}
        for j, (key, lbl) in enumerate(reps):
            cells = []
            vals_all = {}
            for col in bins + ["global"]:
                v = [(r["bins"][col][key][feat] if col != "global" else r["global"][key][feat]) for r in runs]
                v = [x for x in v if x is not None]
                m = _stats.mean(v)
                sd = _stats.stdev(v) if (len(v) > 1 and key == "BeMAE") else 0.0
                vals_all[col] = {"mean": m, "std": sd}
                t = f"{m:.3f} $\\pm$ {sd:.3f}" if key == "BeMAE" else f"{m:.3f}"
                cells.append(t)
            summary[feat][lbl] = vals_all
            # bold the best representation in each column
            lines.append([feat if j == 0 else "", lbl, cells, key])
        if summary[feat]["BeMAE"]["global"]["mean"] > 0:
            summary[feat]["_bemae_low_over_global"] = (
                summary[feat]["BeMAE"]["low"]["mean"] / summary[feat]["BeMAE"]["global"]["mean"])
    # render (bold the highest mean per feature and column)
    out = []
    k = 0
    for feat in features:
        rows = lines[k:k + len(reps)]
        k += len(reps)
        ncol = len(bins) + 1
        best = []
        for c in range(ncol):
            col = bins[c] if c < len(bins) else "global"
            best.append(max(reps, key=lambda r: summary[feat][r[1]][col]["mean"])[1])
        for name, lbl, cells, _ in rows:
            cols = bins + ["global"]
            cells = [f"\\textbf{{{t}}}"
                     if (best[c] == lbl and summary[feat][lbl][cols[c]]["mean"] > 0.05) else t
                     for c, t in enumerate(cells)]
            first = f"\\texttt{{{_tt(feat)}}}" if name else ""
            out.append(f"  {first:<26} & {lbl:<8} & " + " & ".join(cells) + " \\\\")
        out.append("  \\midrule")
    n = summary["_n"]
    tex = (
        "\\begin{tabular}{llccccc}\n"
        "  \\toprule\n"
        "  Feature & Repr.\\ & low ($<\\!60$) & medium & good & excellent ($\\geq\\!300$) & all \\\\\n"
        f"   &  & $n\\!=\\!{_n(n['low'])}$ & ${_n(n['medium'])}$ & ${_n(n['good'])}$ & ${_n(n['excellent'])}$ & \\\\\n"
        "  \\midrule\n" + "\n".join(out[:-1]) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        f"% R^2 of the out-of-fold SpecProbe predictions per SNR bin; BeMAE: mean +- sample std over "
        f"{len(runs)} encoder seed(s); CV = {runs[0].get('cv')}.\n"
    )
    return tex, summary


# ─── T3 EWForecast ──────────────────────────────────────────────────────────
def _fmt_ratio(ratio: float, ci) -> str:
    """Bold when the CI95 of the MAE ratio lies strictly below 1."""
    t = f"{ratio:.3f}"
    return f"\\textbf{{{t}}}" if (ci is not None and ci[1] < 1.0) else t


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else "--"


def _median_abs(values) -> float:
    return float(_stats.median(abs(float(v)) for v in values))


def table_t3_baselines() -> tuple[str, dict]:
    d = load_json(DOWNSTREAM_DIR / "t3_pca_ridge_temporal_baseline.json")
    opt = lambda name: (load_json(DOWNSTREAM_DIR / name) if (DOWNSTREAM_DIR / name).exists() else {})
    d_tsfm = opt("t3_ts_fm_baselines.json")
    cis_path = ROOT / "reference_results" / "stats" / "t3_bootstrap_cis.json"
    cis = load_json(cis_path) if cis_path.exists() else {}
    ci_pca = cis.get("pca_ridge_bootstrap", {}).get("by_config", {})
    ci_fm = cis.get("ts_fm_bootstrap", {}).get("by_config", {})

    def ci_of(table, key):
        c = table.get(key)
        return [c["ratio_mae_ci_lo"], c["ratio_mae_ci_hi"]] if c else None

    summary, rows = {}, []          # rows: list of (block_title | None, [tex lines])

    def add(key, label, mae, mae_p, ratio, ci, r2, med, extra=None):
        rel = mae / med if (med and mae is not None) else None
        summary[key] = {"mae_model": mae, "mae_persist": mae_p, "ratio_mae": ratio,
                        "ratio_mae_ci95": ci, "r2_model": r2,
                        "median_abs_ew": med, "relative_error": rel}
        if extra:
            summary[key].update(extra)
        return (f"  {label} & {mae:.3f} & {mae_p:.3f} & {_fmt_ratio(ratio, ci)} & {_fmt_ci(ci)} & "
                + (f"{r2:.3f}" if r2 is not None else "--") + " & "
                + (f"{100 * rel:.1f}\\%" if rel is not None else "--") + " \\\\")

    # ── trained in-domain: ridge heads on PCA / BeMAE representations ──
    med = {}
    block = []
    for cfg_name in ["pca10_ctx5", "pca10_ctx10", "pca50_ctx5", "pca50_ctx10"]:
        g, c = d[cfg_name]["global"], d[cfg_name]["config"]
        med[c["context_len"]] = _median_abs(d[cfg_name]["predictions"]["y_true"])
        block.append(add(cfg_name, f"PCA({c['n_pca']})$+$ridge, ctx$=${c['context_len']}",
                         g["mae_model"], g["mae_persist"], g["ratio_mae"], ci_of(ci_pca, cfg_name),
                         g["r2_model"], med[c["context_len"]],
                         {"n_pairs": c["n_test"], "n_stars": g.get("n_test_stars")}))
    for ctx in (5, 10):
        zp = DOWNSTREAM_DIR / f"t3_z_ridge_temporal_baseline_ctx{ctx}.json"
        if not zp.exists():
            continue
        dz = load_json(zp)
        gz = dz["per_seed"][str(dz["seeds"][0])]["global"]
        block.append(add(f"z128_ctx{ctx}", f"BeMAE $z_{{128}}$$+$ridge, ctx$=${ctx}",
                         gz["mae_model"], gz["mae_persist"], gz["ratio_mae"], gz.get("ratio_mae_ci95"),
                         gz["r2_model"], med.get(ctx), {"n_stars": gz.get("n_test_stars")}))
    n5 = d["pca10_ctx5"]["config"]["n_test"]
    s5 = d["pca10_ctx5"]["global"].get("n_test_stars")
    rows.append((f"Trained in-domain: ridge head on past representations "
                 f"(ctx$=$5: {_n(n5)} pairs, {s5} stars)", block))

    # ── zero-shot forecasters ──
    block = []
    med_fm = None
    n_fm = s_fm = None
    for key, label in [("chronos_bolt_small", "Chronos-Bolt-S (48\\,M)"),
                       ("chronos_bolt_base",  "Chronos-Bolt-B (205\\,M)"),
                       ("timesfm_2_0_500m",   "TimesFM-2.0 (500\\,M)")]:
        if key not in d_tsfm:
            continue
        g, c = d_tsfm[key]["global"], d_tsfm[key]["config"]
        pr = d_tsfm[key].get("predictions", {})
        if med_fm is None and pr.get("y_true"):
            med_fm = _median_abs(pr["y_true"])
            n_fm, s_fm = c.get("n_test"), len(set(pr.get("star_names", [])))
        block.append(add(key, label, g["mae_model"], g["mae_persist"], g["ratio_mae"],
                         ci_of(ci_fm, key), None, med_fm, {"params_M": c.get("params_M")}))
    extra_fm = [
        ("t3_moirai_zeroshot.json",    "moirai_1.1_R_base",      "Moirai-1.1-R-base (91\\,M)"),
        ("t3_moirai_zeroshot.json",    "moirai_1.1_R_large",     "Moirai-1.1-R-large (311\\,M)"),
        ("t3_moiraimoe_zeroshot.json", "moirai_moe_1.0_R_small", "Moirai-MoE-1.0-R-small (117\\,M)"),
        ("t3_moiraimoe_zeroshot.json", "moirai_moe_1.0_R_base",  "Moirai-MoE-1.0-R-base (935\\,M)"),
        ("t3_timemoe_zeroshot.json",   "time_moe_50M",           "Time-MoE-50M (113\\,M)"),
        ("t3_timemoe_zeroshot.json",   "time_moe_200M",          "Time-MoE-200M (453\\,M)"),
    ]
    for fname, key, label in extra_fm:
        dd = opt(fname)
        if key not in dd:
            continue
        m = dd[key]
        block.append(add(key, label, m["mae"], dd["mae_persist"], m["ratio"], m.get("ci95"),
                         None, med_fm))
    if block:
        title = "Zero-shot time-series foundation models"
        if n_fm:
            title += f" ({_n(n_fm)} pairs, {s_fm} stars)"
        rows.append((title, block))

    # ── trained linear head on a frozen foundation model ──
    block = []
    th, sp = opt("t3_chronos_trained_head.json"), opt("t3_chronos_zeroshot_samepairs.json")
    if th:
        h2, h3 = th["H2_chronos_plus_ctx5"], th["H3_ctx5_only_control"]
        block.append(add("chronos_bolt_base_trained_head", "Chronos-Bolt-B $+$ trained linear head",
                         h2["mae"], h2["mae_persist_same_pairs"], h2["ratio"], h2["ci95"], None, med.get(5),
                         {"n_pairs": h2["n_test"]}))
        block.append(add("ew_context_ridge_control", "\\quad control: ridge on the 5-epoch EW context",
                         h3["mae"], h3["mae_persist_same_pairs"], h3["ratio"], h3["ci95"], None, med.get(5),
                         {"n_pairs": h3["n_test"]}))
    if sp:
        z = sp["ctx5_subset_same_pairs_as_stacking_head"]
        block.append(add("chronos_bolt_base_zeroshot_same_pairs", "\\quad Chronos-Bolt-B zero-shot, same pairs",
                         z["mae"], z["mae_persist"], z["ratio"], z["ci95"], None, med.get(5),
                         {"n_pairs": z["n_test"]}))
    if block:
        rows.append((f"Trained head on a frozen foundation model (the {_n(n5)} ctx$=$5 pairs)", block))

    # ── causal Transformer head on BeMAE embeddings (own evaluation harness) ──
    tf = opt("t3_transformer_head.json")
    if tf:
        ps = list(tf["per_seed"].values())
        ratios = [r["ratio_mae"] for r in ps]
        mae = _stats.mean(r["mae_model"] for r in ps)
        mae_p = ps[0]["mae_persist"]
        r2 = _stats.mean(r["r2_model"] for r in ps)
        mean, sd = _stats.mean(ratios), _stats.pstdev(ratios)
        summary["transformer_head"] = {
            "mae_model": mae, "mae_persist": mae_p, "ratio_mae": mean, "ratio_mae_std": sd,
            "ratio_mae_per_seed": ratios, "r2_model": r2,
            "n_pairs": ps[0]["n_predictions"], "n_stars": ps[0]["n_stars"]}
        rows.append((f"Causal Transformer head on BeMAE embeddings$^{{\\ddagger}}$ "
                     f"({_n(ps[0]['n_predictions'])} pairs, {ps[0]['n_stars']} stars)",
                     [f"  BeMAE $z_{{128}}$ $+$ causal Transformer ({len(ps)} seeds) & {mae:.3f} & {mae_p:.3f} & "
                      f"{mean:.2f} $\\pm$ {sd:.2f} & -- & {r2:.3f} & -- \\\\"]))

    body = []
    for title, block in rows:
        body.append(f"  \\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
        body.extend(block)
        body.append("  \\midrule")
    tex = (
        "\\begin{tabular}{lrrrcrr}\n"
        "  \\toprule\n"
        "  Baseline & MAE & MAE\\textsubscript{persist} & Ratio & CI$_{95}$ of ratio & $R^2$ & Rel.\\ err. \\\\\n"
        "  \\midrule\n"
        + "\n".join(body[:-1]) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        "% MAE in Angstrom; Ratio = MAE / MAE_persist on the same pairs; bold = CI95 strictly below 1; "
        "Rel. err. = MAE / median |EW| of the test targets.\n"
    )
    summary["_median_abs_ew"] = {"ctx5": med.get(5), "ctx10": med.get(10), "zero_shot": med_fm}
    return tex, summary


def table_t3_well_sampled() -> tuple[str, dict]:
    """EWForecast on the well-sampled stars (>= 30 scored epochs) and on the others.

    Reads ``t3_well_sampled_stratification.json`` written by
    ``downstream/t3_well_sampled_stratification.py``.
    """
    path = DOWNSTREAM_DIR / "t3_well_sampled_stratification.json"
    if not path.exists():
        return "", {}
    d = load_json(path)
    labels = [
        ("pca10_ctx5", "PCA(10)$+$ridge, ctx$=$5"),
        ("pca10_ctx10", "PCA(10)$+$ridge, ctx$=$10"),
        ("pca50_ctx5", "PCA(50)$+$ridge, ctx$=$5"),
        ("pca50_ctx10", "PCA(50)$+$ridge, ctx$=$10"),
        ("z128_ctx5", "BeMAE $z_{128}$$+$ridge, ctx$=$5"),
        ("z128_ctx10", "BeMAE $z_{128}$$+$ridge, ctx$=$10"),
        ("chronos_bolt_small", "Chronos-Bolt-S, zero-shot"),
        ("chronos_bolt_base", "Chronos-Bolt-B, zero-shot"),
        ("timesfm_2_0_500m", "TimesFM-2.0, zero-shot"),
    ]
    n_well = d["n_well_sampled_stars"]
    lines = [
        r"\begin{tabular}{lrcc@{\hskip 1.2em}rcc}",
        r"  \toprule",
        rf"  & \multicolumn{{3}}{{c}}{{{n_well} stars with $\geq\!{d['min_epochs']}$ epochs}}"
        r" & \multicolumn{3}{c}{Other stars} \\",
        r"  \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"  Baseline & Pairs & Ratio & CI$_{95}$ & Pairs & Ratio & CI$_{95}$ \\",
        r"  \midrule",
    ]
    summary = {"_n_well_sampled_stars": n_well, "_min_epochs": d["min_epochs"]}
    for key, label in labels:
        if key not in d["baselines"]:
            continue
        b = d["baselines"][key]
        w, o = b["well_sampled"], b["others"]
        summary[key] = {"well_sampled": w, "others": o}
        lines.append(
            f"  {label} & {_n(w['n_pairs'])} & {_fmt_ratio(w['ratio_mae'], w['ratio_mae_ci95'])} & "
            f"{_fmt_ci(w['ratio_mae_ci95'])} & {_n(o['n_pairs'])} & "
            f"{_fmt_ratio(o['ratio_mae'], o['ratio_mae_ci95'])} & {_fmt_ci(o['ratio_mae_ci95'])} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}",
              "% Ratio = MAE / MAE_persist on the pairs of each group; bold = CI95 strictly below 1."]
    return "\n".join(lines) + "\n", summary


def main():
    bundles = {}
    for name, fn in [
        ("stage1",          table_stage1),
        ("t1_probes",       table_t1_probes),
        ("t1_full",         table_t1_full),
        ("t2_linetransfer", table_t2_linetransfer),
        ("snr",             table_snr),
        ("t3_baselines",    table_t3_baselines),
        ("t3_well_sampled", table_t3_well_sampled),
    ]:
        tex, data = fn()
        if not tex:
            print(f"[-] table_{name}: inputs missing, skipped")
            continue
        out_tex = OUT_TABLES / f"table_{name}.tex"
        out_tex.write_text(tex)
        print(f"[+] {out_tex}")
        bundles[name] = data

    # Dump a machine-readable summary so the paper and the figures can pull
    # from a single source of truth.
    summary_path = OUT_TABLES / "summary.json"
    summary_path.write_text(json.dumps(bundles, indent=2))
    print(f"[+] {summary_path}")

    print("\n[=] Headline numbers:")
    t3b = {k: v for k, v in bundles["t3_baselines"].items() if not k.startswith("_")}
    best = min(t3b.items(), key=lambda kv: kv[1]["ratio_mae"])
    print(f"  Lowest EWForecast MAE ratio : {best[0]} -> {best[1]['ratio_mae']:.3f}")


if __name__ == "__main__":
    main()
