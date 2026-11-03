#!/usr/bin/env python3
"""
check_step.py — Verify the results of each export step.

Usage:
    python check_step.py 1    # After extraction → check the Parquet shards
    python check_step.py 2    # After assembly   → check the DatasetDict
    python check_step.py 3    # After push       → check the HuggingFace repo
"""

import os
import json
import sys
from pathlib import Path

EXPORT_DIR = Path(__file__).parent
PARQUET_DIR = Path(os.environ.get("BESS_PARQUET_DIR", "./parquet_spectra"))
HF_DATASET_DIR = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset"))


def check_step_1():
    """Check the FITS → Parquet extraction."""
    print("=== Verification — step 1: extraction ===\n")

    # Extraction report
    report_path = PARQUET_DIR / "extraction_report.json"
    if not report_path.exists():
        print("✗ Report not found:", report_path)
        return False

    with open(report_path) as f:
        report = json.load(f)

    total = report["total_spectra"]
    ok = report["extracted_ok"]
    errors = report["errors"]
    n_shards = report["n_shards"]
    rate = report.get("rate_per_second", 0)

    print(f"  Spectra processed: {total:,}")
    print(f"  Extracted OK     : {ok:,} ({100*ok/total:.1f}%)")
    print(f"  Errors           : {errors:,} ({100*errors/total:.1f}%)")
    print(f"  Parquet shards   : {n_shards}")
    print(f"  Throughput       : {rate:.0f} spectra/s")

    if report.get("error_types"):
        print(f"\n  Error types:")
        for etype, count in sorted(report["error_types"].items(), key=lambda x: -x[1]):
            print(f"    {etype:<30} {count:>6,}")

    # Inspect Parquet files
    parquet_files = sorted(PARQUET_DIR.glob("spectra_*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)

    print(f"\n  Parquet files : {len(parquet_files)}")
    print(f"  Total size    : {total_size / 1e9:.2f} GB")

    # Sample from the first shard
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_files[0])
        print(f"\n  Overview shard 0 : {table.num_rows:,} rows, columns = {table.column_names}")

        row = table.to_pydict()
        wave_len = len(row["wavelength"][0]) if row["wavelength"][0] else 0
        flux_len = len(row["flux"][0]) if row["flux"][0] else 0
        print(f"  First spectrum   : {wave_len} wavelength pixels, {flux_len} flux pixels")
    except Exception as e:
        print(f"  ⚠ Parquet read error: {e}")

    success = errors / total < 0.05 if total > 0 else False
    print(f"\n{'✓' if success else '⚠'} Error rate: {100*errors/total:.1f}% {'(< 5% OK)' if success else '(> 5%, please review)'}")
    return success


def check_step_2():
    """Check the HuggingFace DatasetDict."""
    print("=== Verification — step 2: DatasetDict ===\n")

    if not HF_DATASET_DIR.exists():
        print("✗ Directory not found:", HF_DATASET_DIR)
        return False

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(HF_DATASET_DIR))

        print(f"  Splits:")
        total = 0
        for split, dataset in ds.items():
            n = len(dataset)
            total += n
            print(f"    {split:<12} {n:>8,} spectra")
        print(f"    {'total':<12} {total:>8,}")

        # Schema
        features = ds["train"].features
        print(f"\n  Columns ({len(features)}):")
        for col, feat in features.items():
            print(f"    {col:<35} {feat}")

        # Sample
        sample = ds["train"][0]
        wave = sample.get("wavelength")
        star = sample.get("star_name", "?")
        print(f"\n  Sample train[0]: {star}, {len(wave) if wave else 0} pixels")

        # Disk size
        total_size = sum(f.stat().st_size for f in HF_DATASET_DIR.rglob("*") if f.is_file())
        print(f"  Disk size      : {total_size / 1e9:.2f} GB")

        print(f"\n✓ DatasetDict valid")
        return True

    except Exception as e:
        print(f"✗ Loading error: {e}")
        return False


def check_step_3():
    """Check the HuggingFace push."""
    print("=== Verification — step 3: HuggingFace Hub ===\n")

    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠ HF_TOKEN not set — verification will be limited")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        repo_id = os.environ.get("HF_REPO_ID")
        if not repo_id:
            print("⚠ HF_REPO_ID not set in environment")
            return False

        info = api.dataset_info(repo_id)
        print(f"  Repo         : {repo_id}")
        print(f"  Private      : {info.private}")
        print(f"  Last modified: {info.last_modified}")

        files = api.list_repo_files(repo_id, repo_type="dataset")
        print(f"  Files        : {len(files)}")
        for f in files[:10]:
            print(f"    {f}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")

        print(f"\n✓ Dataset published at https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3"):
        print(__doc__)
        sys.exit(1)

    step = sys.argv[1]
    checkers = {"1": check_step_1, "2": check_step_2, "3": check_step_3}
    success = checkers[step]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
