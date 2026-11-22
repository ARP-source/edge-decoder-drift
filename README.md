# Edge Decoder Drift

If you compress a brain-computer interface decoder so it can run on a small chip, does it get worse at handling the way neural signals change from day to day? And can you still recalibrate it on that chip?

## Why I'm asking this

An implanted BCI decoder has to do two hard things at once. It has to run on very little power, because it lives on the body. And it has to keep working as the recorded neurons shift across days and weeks, without the user recalibrating constantly.

From what I've read, these get studied separately. Drift and robustness papers usually run full precision models on a workstation GPU. Efficiency papers usually test within a single session and estimate power from operation counts instead of measuring it. A real device deals with both at the same time, so I want to measure how they interact.

## Questions

1. Does the gap between same-day accuracy and later-session accuracy grow as the decoder is compressed more?
2. After quantization, does few-shot recalibration still recover performance when the recalibration itself runs on the embedded device?
3. What does the accuracy, latency and energy tradeoff actually look like on real embedded hardware?

Either answer to question 1 is useful. If compression doesn't make drift worse, that's a case for compressing more aggressively.

## Setup

- **Data:** the [FALCON benchmark](https://github.com/snel-repo/falcon-challenge) (Karpowicz et al., NeurIPS 2024 Datasets and Benchmarks). It has multi-session iBCI recordings with held-in and held-out sessions and small calibration sets for recalibration.
- **Hardware:** NVIDIA Jetson Orin Nano Super (8 GB), JetPack 6.2.1, CUDA 12.6. Details in [docs/hardware.md](docs/hardware.md).
- **Energy:** read from the board's INA3221 power rails, so energy per inference is measured, not estimated from FLOPs. These rails cover the module (SoC and memory), not the whole board.
- **Compression levels:** FP32, FP16, INT8 post-training quantization, INT8 quantization-aware training, and structured pruning.
- **Model size:** small enough to train and recalibrate on the device, since that's the setting that matters for deployment.

## Progress

I've set up the hardware, reproduced the published baseline, and (while I don't have the Jetson with me) run the CPU side of questions 1 and 2 on FALCON H1. The hardware parts of the project, Jetson latency and INA3221 energy, are not done yet.

Reproduced FALCON H1 sklearn baseline ([docs/reproduction.md](docs/reproduction.md)):

| Metric | FALCON reported | My run |
|---|---|---|
| Held-in R² (minival) | 0.195 | 0.1943 |
| CV fit score | 0.26 | 0.24 |

The evaluation number matches. The CV gap is small and I haven't tracked it down yet; my best guesses are a scikit-learn version difference or a revised dataset on DANDI.

- [x] Jetson environment set up and recorded
- [x] FALCON H1 baseline reproduced
- [x] PyTorch decoder baseline (H1, CPU)
- [ ] M1 and M2 datasets
- [x] Quantization and pruning sweep (H1, CPU only)
- [x] Few-shot recalibration after compression (H1, CPU only, not on device)
- [ ] Jetson latency measurements
- [ ] Power measurement harness (INA3221)
- [ ] On-device recalibration experiments

### CPU results on FALCON H1 (questions 1 and 2, no hardware)

Decoder: a small causal temporal conv net (3 dilated conv layers, 64 channels, 580 ms receptive field, 98k parameters) on 20 ms binned spikes, trained on the held-in calibration sessions. R² is FALCON's own metric (variance weighted R² per session from `FalconEvaluator.compute_metrics_regression`, then the mean over sessions). Everything is mean ± std over 3 seeds.

| Level | Size (KB) | Held-in R² | Held-out R² (zero-shot) | Gap | Gap minus FP32 (paired) | Held-out after head recal | Held-out after full recal | CPU ms (proxy) |
|---|---|---|---|---|---|---|---|---|
| FP32 | 386 | 0.878 ± 0.009 | 0.233 ± 0.014 | 0.645 ± 0.020 | +0.000 ± 0.000 | 0.238 ± 0.014 | 0.357 ± 0.010 | 0.34 |
| FP16 | 194 | 0.878 ± 0.009 | 0.233 ± 0.014 | 0.645 ± 0.020 | -0.000 ± 0.000 | 0.238 ± 0.014 | 0.357 ± 0.010 | 0.75 |
| INT8 PTQ | 107 | 0.877 ± 0.009 | 0.237 ± 0.015 | 0.640 ± 0.021 | -0.006 ± 0.002 | 0.236 ± 0.015 | 0.356 ± 0.009 | 1.27 |
| INT8 QAT | 107 | 0.879 ± 0.007 | 0.232 ± 0.011 | 0.647 ± 0.016 | +0.002 ± 0.005 | 0.236 ± 0.015 | 0.350 ± 0.010 | 1.25 |
| Prune 50% | 154 | 0.855 ± 0.009 | 0.219 ± 0.001 | 0.635 ± 0.008 | -0.010 ± 0.015 | 0.230 ± 0.003 | 0.352 ± 0.010 | 0.22 |
| Prune 75% | 69 | 0.684 ± 0.013 | 0.214 ± 0.006 | 0.470 ± 0.012 | -0.175 ± 0.023 | 0.220 ± 0.011 | 0.329 ± 0.008 | 0.16 |

What the columns mean:

- **Held-in R²:** FALCON's public held-in minival files, the same split my sklearn reproduction scored 0.194 on.
- **Held-out R² (zero-shot):** FALCON's real held-out evaluation files are private and only scored on their server, so I split each public held-out calibration session in time. The second half is the held-out test set. The decoder never sees held-out data before this number.
- **Gap:** held-in minus held-out. "Gap minus FP32" is the difference from the FP32 gap of the same seed.
- **Recal:** few-shot recalibration on the first half of each held-out calibration session (about 5.7 minutes of data pooled over 14 sessions, roughly 25 seconds per session), scored on the same second halves. "Head" fine-tunes only the output layer, "full" fine-tunes everything. For FP16 and INT8 PTQ I fine-tune the FP32 weights and then cast or requantize. For INT8 QAT I fine-tune the fake-quantized model and convert. Pruned models are fine-tuned as is.
- **Size:** saved state dict. **CPU ms:** median time for one streaming step (one 29 bin window in, one bin out), batch 1, one thread, on the sandbox CPU (ARM Neoverse-V2). This is a rough proxy only and says nothing about the Jetson. The INT8 rows were run with PyTorch's x86 quantization backend, which isn't the native path on ARM; redoing INT8 PTQ for one seed with the ARM (qnnpack) engine gave 0.25 ms vs 0.34 ms for FP32 ([results/h1_latency_qnnpack.json](results/h1_latency_qnnpack.json)).

What I take from this so far:

1. **Question 1 (on CPU, H1 only): quantization doesn't change the drift gap.** FP16 is identical to FP32, and INT8 PTQ and QAT move the gap by less than 0.01 R², which is within the seed to seed spread. Structured pruning does change the gap, but in the opposite direction from what I expected: removing 75% of the channels costs about 0.19 R² held-in but only about 0.02 R² held-out, so the gap shrinks because same-day accuracy drops, not because the model generalizes better. Held-out R² itself still goes down a little with pruning.
2. **Question 2 (simulated on CPU, not on device): few-shot recalibration recovers about the same amount at every compression level.** Full fine-tuning raises held-out R² by about 0.12 to 0.13 at every level. Fine-tuning only the output layer barely helps (under 0.012), which fits the idea that most of the drift is in how the inputs map to the hidden features, not in the readout. Even after full recal, held-out R² (about 0.35) is far below held-in (about 0.88).

Limits:

- CPU only, one dataset (H1), one model family, 3 seeds. No M1 or M2 yet.
- My held-out test set is half of each public held-out calibration session, not FALCON's private held-out eval data, so these held-out numbers aren't comparable to the FALCON leaderboard.
- The pruned models got 600 extra fine-tuning steps on held-in data after pruning; the other levels didn't.
- The held-in number comes from FALCON's minival files. My own validation split (last 10% of each held-in calibration session) scored noticeably lower in early test runs, and I haven't worked out why yet.
- Recalibration ran on a server CPU in FP32 (or with fake quantization for QAT). Whether it works within the Jetson's memory and power budget is still question 2 proper.
- No latency or energy claims about embedded hardware.

Method details, file layout and how to rerun: [docs/cpu_experiments.md](docs/cpu_experiments.md). Figures: [gap vs compression](figures/h1_gap_vs_compression.png), [recalibration recovery](figures/h1_recalibration_recovery.png).

Reading notes are in [docs/literature.md](docs/literature.md).

## Related work

The closest recent paper I know of is FETTUCCINE ([McCart, Deevi et al., NeurIPS 2026](https://arxiv.org/abs/2611.18561)). They run an end-to-end speech brain-to-text decoder on a commercial phone, up to 370 times faster than real time, with word error rates as low as 4.39% on public intracortical speech data. They also show fine-tuning keeps it usable on later days.

That covers part of my question for speech decoding on phone-class hardware. What I want to look at is different in three ways: motor decoding on the FALCON tasks, how much the drift gap changes as the model is compressed further, and energy measured from the power rails instead of throughput alone.

Another relevant paper is NoMAD (Karpowicz et al., [Stabilizing brain-computer interfaces through alignment of latent dynamics](https://www.nature.com/articles/s41467-025-59652-y), Nature Communications 16:4662, 2025). It keeps a decoder stable for weeks to months on monkey motor cortex data by aligning new neural data to a fixed model of latent dynamics, with no labeled recalibration at all. My question 2 is about the other route, few-shot recalibration with a small amount of labeled data, and whether it still works once the model is compressed.

## License

MIT
