"""FALCON H1 compression x drift x recalibration sweep, CPU only.

Work is split into small resumable jobs (each well under two minutes on 2 CPU threads),
so the script can be called repeatedly and picks up where it stopped:

    python scripts/run_h1.py --seeds 0 1 2 --budget 100

Per seed:
  train_fp32   train the FP32 TCN on held-in calib sessions (val = last 10% of each)
  qat          INT8 quantization-aware fine-tuning starting from the FP32 weights
  prune50/75   structured channel pruning (keep 50% / 25% of channels) + held-in fine-tune
  eval         held-in R2 (FALCON minival files) and zero-shot held-out R2 for every level
  recal_*      few-shot recalibration on the first half of each held-out session,
               scored on the second half (head-only and full fine-tune)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import (TCN, Norm, convert_qat, copy_model, cpu_latency_ms, falcon_r2, get_h1,
                    model_bytes, n_params, padded, prepare_qat, prune_channels, ptq_int8, train)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "h1_runs")
CKPT = os.environ.get("CKPT_DIR", "/workspace/work/data/falcon/runs_h1")
LEVELS = ["fp32", "fp16", "int8_ptq", "int8_qat", "prune50", "prune75"]
MODES = ["head", "full"]

CFG = dict(ch=(64, 64, 64), p_drop=0.25, bs=32, T=128, noise=0.1,
           fp32_steps=1500, fp32_lr=1e-3, qat_steps=300, qat_lr=2e-4,
           prune_steps=600, prune_lr=1e-3, recal_steps=300,
           recal_lr={"head": 1e-3, "full": 3e-4}, calib_frac=0.5)

torch.set_num_threads(2)
DATA = None


def data():
    global DATA
    if DATA is None:
        tr, va, ti, rc, to = get_h1(CFG["calib_frac"])
        norm = Norm(tr)
        DATA = dict(train=tr, val=va, test_in=ti, recal=rc, test_out=to, norm=norm,
                    ptrain=padded(tr, norm), precal=padded(rc, norm))
    return DATA


def ck(seed, name):
    d = os.path.join(CKPT, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def new_tcn():
    return TCN(ch=CFG["ch"], p_drop=CFG["p_drop"])


def load_fp32(seed):
    m = new_tcn()
    m.load_state_dict(torch.load(ck(seed, "fp32.pt")))
    return m


def load_qat(seed):
    q = prepare_qat(load_fp32(seed))
    q.load_state_dict(torch.load(ck(seed, "qat.pt")))
    return q


def load_pruned(seed, lvl):
    return torch.load(ck(seed, f"{lvl}.pt"), weights_only=False)


# ------------------------------------------------------------------ jobs


def job_train_fp32(seed):
    D = data()
    torch.manual_seed(seed)
    m = new_tcn()
    part = ck(seed, "fp32_partial.pt")
    stop = None if os.path.exists(part) else CFG["fp32_steps"] // 2  # two chunks
    done = train(m, D["ptrain"], CFG["fp32_steps"], lr=CFG["fp32_lr"], seed=seed, noise=CFG["noise"],
                 val=D["val"], norm=D["norm"], eval_every=250, bs=CFG["bs"], T=CFG["T"],
                 resume=part, stop_at=stop)
    if done:
        torch.save(m.state_dict(), ck(seed, "fp32.pt"))
        os.remove(part)


def job_qat(seed):
    D = data()
    q = prepare_qat(load_fp32(seed))
    train(q, D["ptrain"], CFG["qat_steps"], lr=CFG["qat_lr"], seed=seed + 100, noise=CFG["noise"],
          bs=CFG["bs"], T=CFG["T"])
    torch.save(q.state_dict(), ck(seed, "qat.pt"))


def job_prune(seed, lvl):
    D = data()
    keep = {"prune50": 0.5, "prune75": 0.25}[lvl]
    p = prune_channels(load_fp32(seed), keep)
    train(p, D["ptrain"], CFG["prune_steps"], lr=CFG["prune_lr"], seed=seed + 200, noise=CFG["noise"],
          val=D["val"], norm=D["norm"], eval_every=200, bs=CFG["bs"], T=CFG["T"])
    torch.save(p, ck(seed, f"{lvl}.pt"))


def build_level(seed, lvl, fp32=None):
    """Return (deployable model, dtype) for a compression level, built from checkpoints."""
    D = data()
    if lvl == "fp32":
        return (fp32 or load_fp32(seed)).eval(), torch.float32
    if lvl == "fp16":
        return copy_model(fp32 or load_fp32(seed)).half().eval(), torch.float16
    if lvl == "int8_ptq":
        return ptq_int8(fp32 or load_fp32(seed), D["train"], D["norm"], seed=seed), torch.float32
    if lvl == "int8_qat":
        return convert_qat(load_qat(seed)), torch.float32
    return load_pruned(seed, lvl).eval(), torch.float32


def job_eval(seed):
    D = data()
    out = {}
    for lvl in LEVELS:
        m, dt = build_level(seed, lvl)
        hi = falcon_r2(m, D["test_in"], D["norm"], dt)
        ho = falcon_r2(m, D["test_out"], D["norm"], dt)
        out[lvl] = dict(held_in=hi, held_out=ho, gap=hi["mean"] - ho["mean"],
                        params=n_params(m) if lvl not in ("int8_ptq", "int8_qat") else None,
                        state_dict_bytes=model_bytes(m),
                        cpu_latency_ms_1thread=cpu_latency_ms(m, dt))
    return out


def job_recal(seed, lvl, mode):
    """Few-shot recalibration on the held-out calibration halves.
    FP32 / FP16 / INT8 PTQ: fine-tune the FP32 weights, then re-apply the compression
    (cast to FP16, or re-run PTQ with observers on the recal data).
    INT8 QAT: fine-tune the fake-quant model itself, then convert.
    Pruned: fine-tune the pruned FP32 model."""
    D = data()
    if lvl == "int8_qat":
        base = load_qat(seed)
    elif lvl in ("prune50", "prune75"):
        base = load_pruned(seed, lvl)
    else:
        base = load_fp32(seed)
    m = copy_model(base)
    if mode == "head":
        params = [p for n, p in m.named_parameters() if n.startswith("head")]
        assert params, "head params not found"
        for n, p in m.named_parameters():
            p.requires_grad_(n.startswith("head"))
    else:
        params = list(m.parameters())
    train(m, D["precal"], CFG["recal_steps"], lr=CFG["recal_lr"][mode], seed=seed + 300,
          params=params, noise=CFG["noise"], bs=CFG["bs"], T=CFG["T"])
    for p in m.parameters():
        p.requires_grad_(True)
    if lvl == "fp16":
        dep, dt = copy_model(m).half().eval(), torch.float16
    elif lvl == "int8_ptq":
        dep, dt = ptq_int8(m, D["recal"], D["norm"], seed=seed), torch.float32
    elif lvl == "int8_qat":
        dep, dt = convert_qat(m), torch.float32
    else:
        dep, dt = m.eval(), torch.float32
    ho = falcon_r2(dep, D["test_out"], D["norm"], dt)
    return dict(held_out_recal=ho)


# ------------------------------------------------------------------ runner


def jobs_for(seed):
    js = [("train_fp32", lambda: job_train_fp32(seed), ck(seed, "fp32.pt"), 62),
          ("qat", lambda: job_qat(seed), ck(seed, "qat.pt"), 50),
          ("prune50", lambda: job_prune(seed, "prune50"), ck(seed, "prune50.pt"), 40),
          ("prune75", lambda: job_prune(seed, "prune75"), ck(seed, "prune75.pt"), 40),
          ("eval", lambda: job_eval(seed), os.path.join(RES, f"seed{seed}_eval.json"), 40)]
    for lvl in LEVELS:
        for mode in MODES:
            est = 40 if lvl == "int8_qat" else 25
            js.append((f"recal_{lvl}_{mode}", (lambda l=lvl, mo=mode: job_recal(seed, l, mo)),
                       os.path.join(RES, f"seed{seed}_recal_{lvl}_{mode}.json"), est))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=float, default=100)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    for seed in a.seeds:
        for name, fn, path, est in jobs_for(seed):
            if os.path.exists(path):
                continue
            if time.time() - t0 + est > a.budget:
                print("budget reached, rerun to continue")
                return
            lock = path + ".lock"  # lets two copies of this script share the job list safely
            if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 150:
                continue
            open(lock, "w").close()
            t = time.time()
            try:
                r = fn()
            finally:
                if os.path.exists(lock):
                    os.remove(lock)
            if name == "train_fp32" and not os.path.exists(path):
                print(seed, name, "chunk 1", f"{time.time() - t:.0f}s")
                if time.time() - t0 + est > a.budget:
                    print("budget reached, rerun to continue")
                    return
                t = time.time()
                r = fn()
            if path.endswith(".json"):
                with open(path, "w") as f:
                    json.dump(dict(seed=seed, job=name, cfg={k: v for k, v in CFG.items()}, result=r), f, indent=1)
                print(seed, name, f"{time.time() - t:.0f}s", json.dumps(r)[:200])
            else:
                print(seed, name, f"{time.time() - t:.0f}s")
    print("ALL DONE")


if __name__ == "__main__":
    main()
