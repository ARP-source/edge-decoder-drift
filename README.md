# Edge Decoder Compression vs. Cross-Session Drift

Investigating whether quantizing iBCI decoders for edge deployment (Jetson Orin Nano)
amplifies degradation from cross-session neural nonstationarity, and whether
few-shot recalibration still works post-quantization.

## Status: setup phase

- Hardware: Jetson Orin Nano Super, JetPack 6.2.1, L4T 36.4.7, CUDA 12.6
- Reproduced FALCON benchmark's documented H1 baseline (github.com/snel-repo/falcon-challenge):
  - sklearn_decoder.py CV fit score: 0.24 (paper: 0.26)
  - sklearn_sample.py Held-In R2 Mean: 0.1943 (paper: 0.195)
- Confirms data pipeline + evaluation harness work correctly before building
  anything novel on top.

## Setup notes

- Docker + GPU passthrough confirmed via dustynv/l4t-pytorch:r36.4.0
- FALCON H1 data (DANDI:000954, ~102MB) via `dandi download DANDI:000954`
- falcon-challenge installed via pip in a venv, separate from the docker path
  used for torch/GPU work

## Next steps

- Reproduce a torch-based FALCON baseline
- Move to M1/M2 (monkey, multi-session) datasets
- Precision sweep (FP32/FP16/INT8) with power measurement via INA3221 sysfs
- Few-shot recalibration test on quantized models
