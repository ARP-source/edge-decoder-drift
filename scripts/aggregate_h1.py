"""Collect per-seed JSONs into results/h1_summary.json and make the two figures."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
LEVELS = ["fp32", "fp16", "int8_ptq", "int8_qat", "prune50", "prune75"]
LABELS = {"fp32": "FP32", "fp16": "FP16", "int8_ptq": "INT8 PTQ", "int8_qat": "INT8 QAT",
          "prune50": "Prune 50%", "prune75": "Prune 75%"}

evals = {}
for fn in sorted(glob.glob(f"{RES}/h1_runs/seed*_eval.json")):
    d = json.load(open(fn))
    evals[d["seed"]] = d["result"]
seeds = sorted(evals)
recal = {}
for fn in glob.glob(f"{RES}/h1_runs/seed*_recal_*.json"):
    d = json.load(open(fn))
    _, lvl_mode = d["job"].split("_", 1)
    lvl, mode = lvl_mode.rsplit("_", 1)
    recal.setdefault((lvl, mode), {})[d["seed"]] = d["result"]["held_out_recal"]["mean"]


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)), values=v.tolist())


summary = dict(dataset="FALCON H1 (DANDI 000954)", seeds=seeds,
               notes=("held_in = FALCON held-in minival files; held_out = second half of each public "
                      "held-out calib session (zero-shot); recal = few-shot fine-tune on first half of "
                      "each held-out calib session, scored on the same second half. std is sample std "
                      "across seeds. CPU latency is a single-thread x86 sandbox proxy, not a Jetson "
                      "number."),
               levels={})
for lvl in LEVELS:
    hi = [evals[s][lvl]["held_in"]["mean"] for s in seeds]
    ho = [evals[s][lvl]["held_out"]["mean"] for s in seeds]
    gap = [a - b for a, b in zip(hi, ho)]
    ratio = [b / a for a, b in zip(hi, ho)]
    gap_vs_fp32 = [g - (evals[s]["fp32"]["held_in"]["mean"] - evals[s]["fp32"]["held_out"]["mean"])
                   for g, s in zip(gap, seeds)]
    entry = dict(held_in=ms(hi), held_out=ms(ho), gap=ms(gap), held_out_over_held_in=ms(ratio),
                 gap_minus_fp32_paired=ms(gap_vs_fp32),
                 cpu_latency_ms=ms([evals[s][lvl]["cpu_latency_ms_1thread"] for s in seeds]),
                 state_dict_bytes=evals[seeds[0]][lvl]["state_dict_bytes"],
                 params=evals[seeds[0]][lvl]["params"])
    for mode in ["head", "full"]:
        r = [recal[(lvl, mode)][s] for s in seeds]
        entry[f"recal_{mode}"] = dict(held_out=ms(r), gain=ms([a - b for a, b in zip(r, ho)]),
                                      gap_after=ms([a - b for a, b in zip(hi, r)]))
    summary["levels"][lvl] = entry
json.dump(summary, open(f"{RES}/h1_summary.json", "w"), indent=1)

# table for the README
rows = ["| Level | Size (KB) | Held-in R² | Held-out R² (zero-shot) | Gap | Gap minus FP32 (paired) | Held-out after head recal | Held-out after full recal | CPU ms (proxy) |",
        "|---|---|---|---|---|---|---|---|---|"]
for lvl in LEVELS:
    e = summary["levels"][lvl]
    f = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"
    rows.append(f"| {LABELS[lvl]} | {e['state_dict_bytes'] / 1024:.0f} | {f(e['held_in'])} | {f(e['held_out'])} | "
                f"{f(e['gap'])} | {e['gap_minus_fp32_paired']['mean']:+.3f} ± {e['gap_minus_fp32_paired']['std']:.3f} | "
                f"{f(e['recal_head']['held_out'])} | {f(e['recal_full']['held_out'])} | {e['cpu_latency_ms']['mean']:.2f} |")
open(f"{RES}/h1_table.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))

# figures
os.makedirs(FIG, exist_ok=True)
x = np.arange(len(LEVELS))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, lab, mk in [("held_in", "Held-in (minival)", "o"), ("held_out", "Held-out (zero-shot)", "s")]:
    m = [summary["levels"][l][key]["mean"] for l in LEVELS]
    s = [summary["levels"][l][key]["std"] for l in LEVELS]
    ax[0].errorbar(x, m, yerr=s, marker=mk, capsize=3, label=lab)
ax[0].set_ylabel("R² (FALCON metric)")
ax[0].set_title("H1 accuracy by compression level")
ax[0].legend()
for i, l in enumerate(LEVELS):
    vals = summary["levels"][l]["gap"]["values"]
    ax[1].scatter([i] * len(vals), vals, color="gray", s=12, zorder=3)
ax[1].bar(x, [summary["levels"][l]["gap"]["mean"] for l in LEVELS],
          yerr=[summary["levels"][l]["gap"]["std"] for l in LEVELS], capsize=3, color="tab:blue", alpha=0.6)
ax[1].set_ylabel("Held-in R² minus held-out R²")
ax[1].set_title("Drift gap (dots = seeds)")
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels([LABELS[l] for l in LEVELS], rotation=30)
fig.text(0.5, -0.04, f"FALCON H1, causal TCN, {len(seeds)} seeds, CPU only. Error bars = std across seeds.",
         ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/h1_gap_vs_compression.png", dpi=130, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(7, 4))
w = 0.27
for j, (key, lab) in enumerate([(None, "Zero-shot"), ("recal_head", "Head-only recal"), ("recal_full", "Full recal")]):
    if key is None:
        m = [summary["levels"][l]["held_out"]["mean"] for l in LEVELS]
        s = [summary["levels"][l]["held_out"]["std"] for l in LEVELS]
    else:
        m = [summary["levels"][l][key]["held_out"]["mean"] for l in LEVELS]
        s = [summary["levels"][l][key]["held_out"]["std"] for l in LEVELS]
    ax.bar(x + (j - 1) * w, m, w, yerr=s, capsize=2, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[l] for l in LEVELS], rotation=30)
ax.set_ylabel("Held-out R² (second half of each session)")
ax.set_title("H1 few-shot recalibration after compression")
ax.set_ylim(0, 0.46)
ax.legend(loc="upper center", ncol=3, fontsize=9)
fig.text(0.5, -0.04, f"Recal data = first half of each held-out calib session. {len(seeds)} seeds, CPU only.",
         ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/h1_recalibration_recovery.png", dpi=130, bbox_inches="tight")
print("seeds", seeds)
