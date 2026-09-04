# Hardware baseline

Recorded before any benchmarking, so measurements are attributable to a known
configuration.

| Property | Value |
|---|---|
| Board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| RAM | 8GB (7.4GB usable, unified) |
| Storage | NVMe, 457GB |
| JetPack | 6.2.1 (`nvidia-jetpack 6.2.1+b38`) |
| L4T | R36.4.7 |
| CUDA | 12.6 |
| Power mode | 25W (nvpmodel mode 0) |
| Cooling | Active fan present |

## Measurement caveats

- The INA3221 rails on this dev kit report **module** power (SoC + memory), not
  whole-board power including peripherals. This will be stated explicitly
  alongside any energy figures.
- `jetson_clocks` must be active during timing runs to prevent DVFS-induced
  variance. Baseline state is `Jetson Clocks: inactive`.
- Sustained load may cause thermal throttling; fan state and thermals to be
  logged alongside benchmark runs.

## GPU container

PyTorch with CUDA via `dustynv/l4t-pytorch:r36.4.0` (compatible across r36.4.x).
See `scripts/run_pytorch.sh`.
