# Compression and Cross-Session Drift in Intracortical Decoders

Does compressing an iBCI decoder for on-body deployment make it more vulnerable
to cross-session neural nonstationarity — and can a quantized decoder still be
recalibrated on-device?

## Motivation

Intracortical BCI decoders face two constraints simultaneously in deployment:
they must run on-body under tight power budgets, and they must survive neural
signal drift over days to weeks without constant recalibration.

These constraints are studied separately. Robustness work evaluates at full
precision on workstation GPUs. Efficiency work evaluates within-session and
typically estimates power analytically (MAC counts, circuit models) rather than
measuring it. No deployed device operates in either regime — it operates in both
at once.

This project measures the interaction.

## Research questions

1. Does the held-in to held-out accuracy gap widen as a function of compression
   level? (i.e. does compression amplify drift-induced degradation)
2. Does few-shot recalibration still recover performance after quantization,
   when the recalibration itself runs on the embedded device?
3. What is the measured accuracy / latency / energy frontier for iBCI decoders
   on embedded hardware?

Both outcomes on (1) are informative: no interaction would characterize the
Pareto frontier and license more aggressive edge deployment.

## Approach

- **Data:** [FALCON benchmark](https://github.com/snel-repo/falcon-challenge)
  (Karpowicz et al., NeurIPS 2024 D&B) — multi-session iBCI datasets with
  held-in/held-out splits and few-shot recalibration data
- **Hardware:** NVIDIA Jetson Orin Nano Super (8GB), JetPack 6.2.1, L4T 36.4.7,
  CUDA 12.6
- **Energy measurement:** onboard INA3221 rails via sysfs — measured joules per
  inference, not FLOP proxies
- **Compression:** FP32 → FP16 → INT8 PTQ → INT8 QAT, plus structured pruning
- **Scope constraint:** decoders small enough to train and recalibrate on-device,
  which is the deployment-relevant regime anyway

## Status

**Setup and validation phase.** No novel results yet.

- [x] Hardware environment verified (see `docs/hardware.md`)
- [x] FALCON H1 baselines reproduced (see `docs/reproduction.md`)
- [ ] Torch decoder baseline reproduced
- [ ] M1/M2 (multi-session monkey) datasets
- [ ] Quantization sweep
- [ ] Power measurement harness
- [ ] On-device recalibration experiments

## Related work

See `docs/literature.md` — survey in progress.

## License

MIT
