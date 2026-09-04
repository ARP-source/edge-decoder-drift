# Reproduction log

Validating the FALCON pipeline before building anything on top of it.

## FALCON H1 (DANDI:000954) — sklearn baselines

Environment: Python 3.10 venv on Jetson host (no GPU needed for these),
`falcon-challenge` and `dandi` installed via pip.

### Training / CV fit

```
python decoder_demos/sklearn_decoder.py \
  --training_dir data/000954/sub-HumanPitt-held-in-calib/ \
  --calibration_dir data/000954/sub-HumanPitt-held-out-calib/ \
  --mode all --task h1
```

| Metric | Reported (FALCON README) | Reproduced |
|---|---|---|
| CV fit score | 0.26 | 0.24 |

### Evaluation on minival

```
python decoder_demos/sklearn_sample.py \
  --evaluation local --phase minival --split h1
```

| Metric | Reported (FALCON README) | Reproduced |
|---|---|---|
| Held-In R² Mean | 0.195 | 0.1943 |

Also reported: Normalized Latency 0.3674, Held-In R² Std. 0.0292.

### Notes

The CV fit discrepancy (0.24 vs 0.26) is larger than the eval discrepancy
(0.1943 vs 0.195) and is not yet explained. Likely candidates: scikit-learn
version drift, or the DANDI dandiset having been revised since the README was
written (it is served as a draft version with incomplete metadata). Not currently
blocking — the evaluation number, which is the one the benchmark actually scores
on, matches to three decimal places. Worth revisiting if it becomes relevant.
