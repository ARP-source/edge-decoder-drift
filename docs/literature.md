# Related work

**Status: in progress.** Entries below are triaged against four criteria:
cross-session evaluation, measured (not estimated) energy, real embedded
hardware, and post-compression recalibration.

| Paper | Cross-session | Measured energy | Real hardware | Recalibration | Signal |
|---|---|---|---|---|---|
| Hueber et al. 2024, *Benchmarking hardware-efficient real-time neural decoding* (Neuromorph. Comput. Eng.) | No | No — MAC/memory-access estimates vs. hypothetical 1MHz clock | No | No | Intracortical (Makin/Dyer) |
| Zhou et al. 2024, *Benchmarking neural decoding backbones for on-edge iBCI* (arXiv 2406.06626) | Partial | No | No | Partial — "new session fine-tuning" | Intracortical NHP |
| Karpowicz et al. 2024, *Reducing power requirements for high-accuracy decoding in iBCIs* (J. Neural Eng.) | No — within-day | Partial — circuit-model calculations (NEF, Schreier FoM) | No | No | Spikes + LFP |
| Mei et al. 2024, *Train-on-Request* (BioCAS) | Yes | Yes — 1mJ/step on GAP9 | Yes | Yes | Scalp EEG |
| Fang et al. 2025, *Lightweight neural decoder design using QAT* (IEEE NER) | **Unread** | **Unread** | Simulation study per abstract | **Unread** | — |

## Reading of the gap

Two patterns across the accessible papers:

1. **Nobody measures joules on hardware for intracortical decoding.** Power is
   estimated from MAC counts (Hueber) or circuit-component formulas (Karpowicz).
   The one paper with real measured energy (Mei) is scalp EEG on a GAP9.
2. **The axes exist separately.** Karpowicz compresses the *signal* for
   transmission; Zhou compares *architectures*; Mei does on-device learning
   *without* compression. None compress the *model* and then ask what drift does
   to it.

## Open

- **Fang et al. 2025 is unread** — TechRxiv preprint behind a Cloudflare wall,
  IEEE version paywalled. This is the closest known work (QAT applied to neural
  decoders, from Orsborn lab at UW) and the claim of an open gap is provisional
  until it is read.
- Leads not yet triaged: Rivelli et al. 2025 (pruned SNNs for intracortical
  decoding, arXiv 2504.11568); Chen et al. 2025 (transfer meta-learning with a
  "fast self-recalibrating decoder", IEEE Trans.).
