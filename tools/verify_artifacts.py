#!/usr/bin/env python3
"""
verify_artifacts.py — Sanity checks for the public BESS-Bench release.

Usage:
    python tools/verify_artifacts.py [RELEASE_DIR] [--final]

With no directory, verifies the current repository. ``--final`` turns the
warnings about review-time identifiers and placeholders into errors.

Returns 0 if everything is OK, 1 otherwise.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

# ── Expected figures (taken from the paper) ─────────────────────────────────
EXPECTED = {
    "n_stars":        1468,
    "n_spectra":      339_115,
    "n_train_stars":  1024,
    "n_val_stars":    219,
    "n_test_stars":   225,
    "n_train_spec":   241_057,
    "n_val_spec":     44_383,
    "n_test_spec":    53_675,
}

LOCAL_PATH_PATTERNS = ["/share/home", "/share/projects", "/home/", "/Users/"]
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")
REVIEW_TIME_PATTERNS = ["anonym-submit-26", "OWNER/", "author    = {Anonymous}",
                        "author = {Anonymous}", '"name": "Anonymous"']

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "REPRODUCE.md",
    "EVALUATION_PROTOCOL.md",
    "requirements.txt",
    "dataset/README.md",
    "dataset/DATASHEET.md",
    "dataset/croissant.json",
    "dataset/splits.csv",
    "dataset/star_list.csv",
    "dataset/known_issues_v1.0.csv",
    "scripts/reproduce_paper_seed42.sh",
    "reference_results/stats/t1_multiseed_summary.json",
    "reference_results/stats/t3_bootstrap_cis.json",
    "reference_results/stage1_encoder/multiseed_summary.json",
    "paper/tables/summary.json",
]

ERRORS = []
WARNINGS = []


def err(msg):
    ERRORS.append(f"  ❌ {msg}")


def warn(msg):
    WARNINGS.append(f"  ⚠️  {msg}")


def ok(msg):
    print(f"  ✅ {msg}")


args = [a for a in sys.argv[1:] if not a.startswith("--")]
FINAL = "--final" in sys.argv[1:]
ROOT = Path(args[0]) if args else Path(__file__).resolve().parent.parent
print(f"\n🔍 Verifying: {ROOT}\n")


# ── 1. Required files ───────────────────────────────────────────────────────
print("── 1. Required files ──────────────────────────────────────────────────")
for rel in REQUIRED_FILES:
    if (ROOT / rel).exists():
        ok(rel)
    else:
        err(f"Missing file: {rel}")


# ── 2. splits.csv — star counts and MD5 formula ─────────────────────────────
print("\n── 2. splits.csv ──────────────────────────────────────────────────────")
splits_path = ROOT / "dataset/splits.csv"
stars_split = {}
if splits_path.exists():
    counts = {}
    with open(splits_path) as f:
        for row in csv.DictReader(f):
            counts[row["split"]] = counts.get(row["split"], 0) + 1
            stars_split[row["star_name"]] = row["split"]

    for split_name, key in [("train", "n_train_stars"), ("validation", "n_val_stars"),
                            ("test", "n_test_stars")]:
        got, exp = counts.get(split_name, 0), EXPECTED[key]
        ok(f"{split_name}: {got} stars") if got == exp else err(
            f"{split_name}: expected {exp}, found {got}")

    total = sum(counts.values())
    ok(f"Total: {total} stars") if total == EXPECTED["n_stars"] else err(
        f"Total: expected {EXPECTED['n_stars']}, found {total}")

    mismatches = 0
    for star, split in stars_split.items():
        h = int(hashlib.md5(f"bess_bench_v1::{star}".encode()).hexdigest()[:8], 16) % 100
        expected_split = "train" if h < 70 else "validation" if h < 85 else "test"
        mismatches += split != expected_split
    ok("MD5 formula reproduces the partition") if mismatches == 0 else err(
        f"MD5 formula: {mismatches}/{total} disagreements")


# ── 3. star_list.csv — spectrum counts and consistency with splits.csv ──────
print("\n── 3. star_list.csv ───────────────────────────────────────────────────")
star_list_path = ROOT / "dataset/star_list.csv"
if star_list_path.exists():
    total_spec, spec_per_split, star_list_splits = 0, {}, {}
    with open(star_list_path) as f:
        for row in csv.DictReader(f):
            n = int(row["n_spectra"])
            total_spec += n
            spec_per_split[row["split"]] = spec_per_split.get(row["split"], 0) + n
            star_list_splits[row["star_name"]] = row["split"]

    ok(f"Total n_spectra: {total_spec}") if total_spec == EXPECTED["n_spectra"] else err(
        f"Total n_spectra: expected {EXPECTED['n_spectra']}, found {total_spec}")
    for split_name, key in [("train", "n_train_spec"), ("validation", "n_val_spec"),
                            ("test", "n_test_spec")]:
        got, exp = spec_per_split.get(split_name, 0), EXPECTED[key]
        ok(f"n_spectra {split_name}: {got}") if got == exp else err(
            f"n_spectra {split_name}: expected {exp}, found {got}")
    if stars_split:
        mismatch = sum(1 for k in stars_split if stars_split[k] != star_list_splits.get(k))
        ok("Partition consistent between splits.csv and star_list.csv") if mismatch == 0 else err(
            f"{mismatch} stars have different splits in splits.csv and star_list.csv")


# ── 4. Local paths, tokens and review-time identifiers ──────────────────────
print("\n── 4. Local paths, tokens, review-time identifiers ────────────────────")
TEXT_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".tex", ".txt",
             ".toml", ".cfg", ".slurm", ".rst", ".cff", ""}
n_leaks, review_hits = 0, {}
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or {".git", ".venv", "__pycache__"} & set(p.parts):
        continue
    if p.resolve() == Path(__file__).resolve():   # this file lists the patterns
        continue
    if p.suffix not in TEXT_EXTS and p.name != "LICENSE":
        continue
    text = p.read_text(errors="ignore")
    rel = p.relative_to(ROOT)
    for pat in LOCAL_PATH_PATTERNS:
        if pat in text:
            err(f"Local path '{pat}' in {rel}")
            n_leaks += 1
            break
    if TOKEN_PATTERN.search(text):
        err(f"Possible access token in {rel}")
        n_leaks += 1
    for pat in REVIEW_TIME_PATTERNS:
        if pat in text:
            review_hits.setdefault(pat, []).append(str(rel))
if n_leaks == 0:
    ok("No local path or token found")
for pat, files in review_hits.items():
    msg = f"'{pat}' still present in {len(files)} file(s), e.g. {files[0]}"
    err(msg) if FINAL else warn(msg)
if not review_hits:
    ok("No review-time identifier or placeholder left")


# ── 5. Internal consistency of the reported numbers ─────────────────────────
print("\n── 5. Reported numbers ────────────────────────────────────────────────")
try:
    multi = json.loads((ROOT / "reference_results/stats/t1_multiseed_summary.json").read_text())
    head = multi["headline_aggregated_ratio"]
    ratio = f"{head['ratio_point']:.2f}"
    ci = f"[{head['ratio_ci95'][0]:.2f}, {head['ratio_ci95'][1]:.2f}]"
    if not str(multi.get("cv", "")).startswith("GroupKFold"):
        err(f"SpecProbe reference results are not star-grouped (cv = {multi.get('cv')})")
    for rel in ("dataset/README.md", "stats/README.md"):
        text = (ROOT / rel).read_text()
        if ratio in text and ci in text:
            ok(f"{rel} quotes the headline ratio {ratio} {ci}")
        else:
            err(f"{rel} does not quote the headline ratio {ratio} {ci}")
    croissant = json.loads((ROOT / "dataset/croissant.json").read_text())
    repo = croissant.get("url", "").split("/datasets/")[-1]
    if repo and repo in (ROOT / "dataset/README.md").read_text():
        ok(f"croissant.json and the dataset card agree on the repository ({repo})")
    else:
        err(f"croissant.json url '{croissant.get('url')}' not found in the dataset card")
except FileNotFoundError as exc:
    err(f"Cannot check the reported numbers: {exc}")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
for w in WARNINGS:
    print(w)
if ERRORS:
    print(f"\n🔴 {len(ERRORS)} error(s):")
    for e in ERRORS:
        print(e)
    print()
    sys.exit(1)
print(" ✅  All checks pass." + ("" if FINAL or not WARNINGS else
                                  "  (warnings above must be cleared before publication)"))
print()
sys.exit(0)
