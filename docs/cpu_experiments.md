# CPU experiments (FALCON H1)

These runs answer the CPU side of questions 1 and 2 while I don't have the Jetson.
Nothing here is a hardware measurement.

## Data

FALCON H1, DANDI 000954 (draft version, public, no account needed), 40 NWB files, about 98 MB.
Binned with `falcon_challenge.dataloaders.load_nwb` (20 ms bins, 176 channels, 7 velocity dims),
so binning and eval masks match FALCON.

| Split | Source | Amount |
|---|---|---|
| Train | first 90% of each held-in calib session (13 sessions) | 40.8 min |
| Validation (checkpoint selection) | last 10% of each held-in calib session | |
| Held-in test | FALCON held-in minival files (13 sessions) | |
| Recalibration set | first half of each held-out calib session (14 sessions) | 5.7 min |
| Held-out test | second half of each held-out calib session | 5.7 min |

FALCON's held-out eval files are private (scored on EvalAI), which is why I carve the held-out
test set out of the public held-out calibration files.

Inputs are z-scored with training statistics (std + 1 to keep near-silent channels stable).
No per-session re-normalization is done, so the zero-shot held-out number is truly zero-shot.

## Model and training

Causal TCN: Conv1d(176, 64, k=5, d=1), Conv1d(64, 64, k=5, d=2), Conv1d(64, 64, k=5, d=4), each with
ReLU (dropout 0.25 after the first two), then a 1x1 conv to 7 outputs. Receptive field 29 bins (580 ms).
97,927 parameters. AdamW, lr 1e-3, cosine schedule, 1500 steps of 32 windows of 128 bins, input
noise 0.1, masked MSE on eval-mask bins, best validation checkpoint kept.

## Compression levels

- FP32: the trained model.
- FP16: `model.half()`, inference in half precision on CPU.
- INT8 PTQ: static post-training quantization with FX graph mode (`prepare_fx`/`convert_fx`, x86 qconfig), observers on 20 batches of training data.
- INT8 QAT: `prepare_qat_fx` from the FP32 weights, 300 steps at lr 2e-4, then `convert_fx`.
- Prune 50% / 75%: structured channel pruning by filter L1 norm in all three hidden layers, channels physically removed (38,727 and 16,807 parameters), then 600 fine-tuning steps on held-in data at lr 1e-3.

`torch.ao.quantization` is deprecated in favor of torchao in this PyTorch version but still works.
The x86 qconfig was used even though the sandbox CPU is ARM; `scripts/latency_qnnpack.py` repeats
INT8 PTQ with the ARM engine for one seed (same accuracy, faster).

## Recalibration

300 steps on the recalibration set. Head-only: only the last 1x1 conv trains (lr 1e-3).
Full: all weights train (lr 3e-4). FP16 and INT8 PTQ fine-tune the FP32 weights and then cast or
requantize (PTQ observers on the recalibration data). INT8 QAT fine-tunes the fake-quantized model.
No early stopping, since that would need labeled held-out data beyond the few-shot set.

## Seeds

Seeds 0, 1, 2. Each seed changes initialization, batch sampling and dropout.

## Files

- `scripts/prep_h1.py`: NWB to per-session npz.
- `scripts/common.py`: data splits, model, training, FALCON metric wrapper, compression helpers, latency.
- `scripts/run_h1.py`: the sweep, split into small resumable jobs (the sandbox kills commands after 2 minutes).
- `scripts/aggregate_h1.py`: builds `results/h1_summary.json`, `results/h1_table.md` and the figures.
- `scripts/latency_qnnpack.py`: ARM INT8 latency check.
- `results/h1_runs/`: one JSON per seed per job, with per-session R².

## Rerun

```
pip install -r requirements.txt
# download DANDI 000954 into $FALCON_DIR (default /workspace/work/data/falcon/000954)
python scripts/prep_h1.py
cd scripts && python run_h1.py --seeds 0 1 2   # repeat until it prints ALL DONE
python aggregate_h1.py
```
