---
license: cc-by-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - feature-extraction
  - time-series-forecasting
tags:
  - astronomy
  - spectroscopy
  - be-stars
  - h-alpha
  - sample
pretty_name: BESS-Bench v1.0 (inspection sample)
---

# BESS-Bench v1.0 — inspection sample

This is a **stratified sample** of the full BESS-Bench v1.0 dataset
[`matthieulel/bess-bench`](https://huggingface.co/datasets/matthieulel/bess-bench),
provided so that the data can be inspected without downloading the
10 GB full release. It is the sample that was made available during the
peer review of the NeurIPS 2026 paper (file unchanged).

```python
from datasets import load_dataset
sample = load_dataset("matthieulel/bess-bench-sample", split="train")
```

## How the sample was built

- **Source.** The full release (339,115 spectra, 35 columns).
- **Star-level splits.** The star-level partition of the full release
  (`splits.csv`, 1,024 train / 219 validation / 225 test stars) is used
  to stratify the draw.
- **Sampling rule (seed 42).** For each of the three splits, draw
  **50 stars uniformly at random**; for every selected star, retain at
  most **20 spectra**. Union over the three splits.
- **Result.** **1,976 spectra** (0.6 % of the corpus) from 150 stars
  (692 train / 685 validation / 599 test rows), with the split
  membership stored in an additional `split` column.

The schema, dtypes, column semantics and metadata conventions are
**identical** to the full release; only the rows are subsampled. No
spectrum is altered. The known issues of v1.0 documented in the card of
the full release apply to the sample as well: 4 of its rows have a
`wavelength` array in nm instead of Å.

## Reproducing the sample

The build script is `scripts/build_review_sample.py` in the
[code repository](https://github.com/matthieulel/bess-bench).

## Differences with the full release

| Property | Full release | This sample |
|---|---|---|
| Repository | `matthieulel/bess-bench` | `matthieulel/bess-bench-sample` |
| Spectra | 339,115 | 1,976 |
| Stars | 1,468 | 150 (50 per split) |
| Per-star cap | none | 20 spectra |
| `split` column | absent (use `splits.csv`) | present |
| Schema | 35 columns | 35 + `split` |

## Licence

CC-BY-4.0, same as the full release. See the card of the full dataset
for attribution and citation guidance.
