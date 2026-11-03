#!/usr/bin/env python3
"""
generate_croissant.py — retarget and check the Croissant descriptor of BESS-Bench.

``dataset/croissant.json`` (Croissant 1.1 with the Responsible-AI and
provenance fields required by the NeurIPS Evaluations & Datasets track) is
the source of truth and is edited by hand. This script

  * rewrites the repository URLs for a given HuggingFace dataset id,
  * refreshes the SHA-256 of the CSV files it describes from the local copies
    in ``dataset/``,
  * validates the result with ``mlcroissant`` when the package is installed.

Usage:
    python stats/generate_croissant.py --check                 # validate only
    python stats/generate_croissant.py --repo-id <owner>/<name>   # retarget in place

Spec : https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html
RAI  : https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
HF_PREFIX = "https://huggingface.co/datasets/"
REQUIRED_RAI = ["rai:dataLimitations", "rai:dataBiases", "rai:personalSensitiveInformation",
                "rai:dataUseCases", "rai:dataSocialImpact", "rai:hasSyntheticData",
                "prov:wasDerivedFrom", "prov:wasGeneratedBy"]


def retarget(text: str, old_repo: str, new_repo: str) -> str:
    """Replace the dataset repository id everywhere in the JSON text."""
    return text.replace(HF_PREFIX + old_repo, HF_PREFIX + new_repo).replace(old_repo, new_repo)


def refresh_checksums(doc: dict, dataset_dir: Path) -> list[str]:
    """Recompute the SHA-256 of the described files that exist locally."""
    changed = []
    for obj in doc.get("distribution", []):
        url = obj.get("contentUrl", "")
        local = dataset_dir / url.rsplit("/", 1)[-1]
        if obj.get("@type") == "cr:FileObject" and "/resolve/" in url and local.is_file():
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            if obj.get("sha256") != digest:
                obj["sha256"] = digest
                changed.append(local.name)
    return changed


def validate(path: Path) -> bool:
    doc = json.loads(path.read_text())
    missing = [k for k in REQUIRED_RAI if k not in doc]
    if missing:
        print(f"missing Responsible-AI / provenance fields: {missing}")
        return False
    try:
        import mlcroissant as mlc
    except ImportError:
        print("mlcroissant is not installed: structural validation skipped "
              "(pip install mlcroissant)")
        return True
    try:
        mlc.Dataset(jsonld=str(path))
    except Exception as exc:  # mlcroissant raises ValidationError with the full report
        print(f"mlcroissant validation failed:\n{exc}")
        return False
    print("mlcroissant validation: OK")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=DATASET_DIR / "croissant.json")
    ap.add_argument("--repo-id", default=None,
                    help="HuggingFace dataset id to write in the descriptor (<owner>/<name>)")
    ap.add_argument("--check", action="store_true", help="validate without writing")
    args = ap.parse_args()

    text = args.file.read_text()
    doc = json.loads(text)
    current = doc["url"].removeprefix(HF_PREFIX)
    if args.repo_id and args.repo_id != current and not args.check:
        doc = json.loads(retarget(text, current, args.repo_id))
        print(f"repository: {current} -> {args.repo_id}")
    changed = refresh_checksums(doc, args.file.parent)
    if changed:
        print(f"checksums {'differ' if args.check else 'refreshed'}: {', '.join(changed)}")
    if not args.check:
        args.file.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"written: {args.file}")
    elif changed:
        return 1
    return 0 if validate(args.file) else 1


if __name__ == "__main__":
    sys.exit(main())
