# stage2_transformer_head/ — causal Transformer head (EWForecast, negative result)

A small causal Transformer that reads the sequence of frozen BeMAE
embeddings of a star and predicts the embedding of the next observation;
EW(Hα) is then read out from the predicted embedding by a ridge regression
fitted after training. It is the "causal Transformer head on BeMAE
embeddings" row of the EWForecast table of the paper: it does **not** beat
persistence (MAE ratio 1.17 ± 0.05 over three head seeds, range 1.11–1.22).

## Configuration of the paper

4 layers, d_model = 128, 4 heads, FFN 256, dropout 0.1 (≈627 k parameters),
prediction of the embedding increment (`--predict_delta`) conditioned on
the time to the next observation (`--condition_dt`), next-embedding loss
only, AdamW (lr 3e-4, weight decay 0.05), up to 150 epochs with early
stopping (patience 30), temporal cutoff MJD 59215, seed-42 BeMAE
embeddings (`reference_results/embeddings/star_data.pt`).

```bash
cd stage2_transformer_head
for s in 42 123 456; do
    python train.py --run_name tfm_seed$s --seed $s \
        --n_layers 4 --dropout 0.1 --predict_delta --condition_dt
    python evaluate.py --checkpoint runs/tfm_seed$s/best.pt
done
```

About 0.6 GPU-hour for the three seeds on one RTX 2080 Ti. `train.py`
imports `wandb`; run with `WANDB_MODE=disabled` for no logging. Recent
PyTorch versions print a deprecation warning about attention-mask types at
every forward pass (`python -W ignore evaluate.py ...` silences it).

The number of the paper is `T0b_ew_tracking.ratio_mae` in
`runs/<run>/eval_results.json` (EW read out by the post-hoc ridge). The
three reference runs are summarised in
`reference_results/downstream/t3_transformer_head.json`:

| Head seed | MAE (Å) | MAE persistence (Å) | Ratio |
|---|---:|---:|---:|
| 42 | 1.883 | 1.602 | 1.175 |
| 123 | 1.775 | 1.602 | 1.108 |
| 456 | 1.955 | 1.602 | 1.220 |

Mean 1.168; the ± 0.05 of the paper is the population standard deviation
over the three seeds (0.046; sample standard deviation 0.056). GPU training
is not bit-reproducible: a new run gives slightly different values.

## What to keep in mind when reading this number

- **Own evaluation population.** The head is scored with its own harness:
  stars with at least 5 epochs and at least 3 post-cutoff epochs, i.e.
  6 495 test pairs from 479 stars (persistence MAE 1.602 Å). This differs
  from the 7 313 pairs / 621 stars of the ridge baselines, so the ratio is
  comparable to theirs only through the common persistence reference.
- **Checkpoint selection uses the test period.** The loader that drives
  early stopping and the choice of `best.pt` is built from the post-cutoff
  observations of the test stars. This can only favour the head, and it
  still does not beat persistence, so the negative conclusion is
  unaffected; a submission claiming to *beat* persistence must select its
  checkpoint on pre-cutoff data only (see `EVALUATION_PROTOCOL.md`).
- **Other heads in the code.** `model.py` also contains a direct EW head, an
  auxiliary EW head and a contrastive loss; they are switched off in the
  configuration of the paper (`alpha_direct_ew = alpha_ew = alpha_contr = 0`),
  which is why the `T0b_direct_ew` entry of the evaluation is meaningless
  for these runs.

## Files

- `config.py` — model and training configuration
- `dataset.py` — per-star embedding sequences, temporal split
- `model.py` — causal Transformer and prediction heads
- `train.py` — training loop
- `evaluate.py` — evaluation (next-embedding prediction, EW tracking, trend, change points)
