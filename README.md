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

Early stage. I've set up the hardware and reproduced the published baseline. No new results yet.

Reproduced FALCON H1 sklearn baseline ([docs/reproduction.md](docs/reproduction.md)):

| Metric | FALCON reported | My run |
|---|---|---|
| Held-in R² (minival) | 0.195 | 0.1943 |
| CV fit score | 0.26 | 0.24 |

The evaluation number matches. The CV gap is small and I haven't tracked it down yet; my best guesses are a scikit-learn version difference or a revised dataset on DANDI.

- [x] Jetson environment set up and recorded
- [x] FALCON H1 baseline reproduced
- [ ] PyTorch decoder baseline
- [ ] M1 and M2 datasets
- [ ] Quantization sweep
- [ ] Power measurement harness
- [ ] On-device recalibration experiments

Reading notes are in [docs/literature.md](docs/literature.md).

## Related work

The closest recent paper I know of is FETTUCCINE ([McCart, Deevi et al., NeurIPS 2026](https://arxiv.org/abs/2611.18561)). They run an end-to-end speech brain-to-text decoder on a commercial phone, up to 370 times faster than real time, with word error rates as low as 4.39% on public intracortical speech data. They also show fine-tuning keeps it usable on later days.

That covers part of my question for speech decoding on phone-class hardware. What I want to look at is different in three ways: motor decoding on the FALCON tasks, how much the drift gap changes as the model is compressed further, and energy measured from the power rails instead of throughput alone.

## License

MIT
