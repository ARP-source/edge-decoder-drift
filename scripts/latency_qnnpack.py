"""Extra latency check. The sandbox CPU turned out to be ARM (Neoverse-V2), and the main sweep
quantized with the x86 (fbgemm) qconfig, which is not the native INT8 path on ARM. This
re-runs INT8 PTQ for seed 0 with the qnnpack (ARM) engine and times it next to FP32.
Still a CPU proxy, not a Jetson number."""
import json, os, platform
import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from common import TCN, Norm, cpu_latency_ms, example_input, falcon_r2, get_h1, padded, sample_batch

torch.set_num_threads(2)
tr, va, ti, rc, to = get_h1()
norm = Norm(tr)
m = TCN(ch=(64, 64, 64), p_drop=0.25)
m.load_state_dict(torch.load("/workspace/work/data/falcon/runs_h1/seed0/fp32.pt"))
m.eval()
torch.backends.quantized.engine = "qnnpack"
pm = prepare_fx(m, get_default_qconfig_mapping("qnnpack"), (example_input(),))
rng = np.random.default_rng(0)
data = padded(tr, norm)
with torch.no_grad():
    for _ in range(20):
        pm(sample_batch(data, rng, bs=32, T=256)[0])
q = convert_fx(pm)
out = dict(cpu=platform.machine(), engine="qnnpack", seed=0,
           fp32_ms=cpu_latency_ms(m), int8_ptq_qnnpack_ms=cpu_latency_ms(q),
           int8_ptq_qnnpack_held_in=falcon_r2(q, ti, norm)["mean"],
           int8_ptq_qnnpack_held_out=falcon_r2(q, to, norm)["mean"])
print(out)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "h1_latency_qnnpack.json"), "w"), indent=1)
