"""Shared pieces: data loading, the causal TCN decoder, training, FALCON-style R2, compression helpers."""
import glob
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from falcon_challenge.evaluator import FalconEvaluator  # noqa: E402

NPZ = os.environ.get("FALCON_NPZ", "/workspace/work/data/falcon/h1_npz")

# ---------------------------------------------------------------- data


def load_split(split):
    out = []
    for fn in sorted(glob.glob(f"{NPZ}/{split}_*.npz")):
        d = np.load(fn)
        ses = os.path.basename(fn)[len(split) + 1:-4]
        out.append(dict(ses=ses, x=d["x"].astype(np.float32), y=d["y"].astype(np.float32),
                        m=d["eval_mask"].astype(bool)))
    return out


def cut(s, a, b):
    """Slice a session dict by fraction [a, b) of its length."""
    n = len(s["x"])
    i, j = int(a * n), int(b * n)
    return dict(ses=s["ses"], x=s["x"][i:j], y=s["y"][i:j], m=s["m"][i:j])


def get_h1(calib_frac=0.5):
    """Held-in train/val from held-in-calib (last 10% of each session is val),
    held-in test = FALCON held-in minival files,
    held-out: the public held-out-calib files, split per session in time.
    The first `calib_frac` of each held-out session is the few-shot recalibration set,
    the rest is the held-out test set. FALCON's real held-out eval files are private
    (scored on their server), so this split is my stand-in."""
    hic = load_split("held-in-calib")
    train = [cut(s, 0.0, 0.9) for s in hic]
    val = [cut(s, 0.9, 1.0) for s in hic]
    test_in = load_split("held-in-minival")
    hoc = load_split("held-out-calib")
    recal = [cut(s, 0.0, calib_frac) for s in hoc]
    test_out = [cut(s, 0.5, 1.0) for s in hoc]  # fixed test half regardless of calib_frac
    return train, val, test_in, recal, test_out


class Norm:
    def __init__(self, sessions):
        x = np.concatenate([s["x"] for s in sessions])
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1.0  # +1 keeps silent channels from blowing up
        y = np.concatenate([s["y"][s["m"]] for s in sessions])
        self.ysd = float(y.std())

    def x(self, x):
        return ((x - self.mu) / self.sd).astype(np.float32)


# ---------------------------------------------------------------- model

KS, DIL = 5, (1, 2, 4)
RF = 1 + sum((KS - 1) * d for d in DIL)  # receptive field in bins (29 bins = 580 ms)


class TCN(nn.Module):
    """Causal temporal conv decoder. No padding inside, so the caller left-pads RF-1 bins.
    That keeps the graph simple for FX quantization."""

    def __init__(self, n_in=176, n_out=7, ch=(128, 128, 128), p_drop=0.25):
        super().__init__()
        c1, c2, c3 = ch
        self.c1 = nn.Conv1d(n_in, c1, KS, dilation=DIL[0])
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(p_drop)
        self.c2 = nn.Conv1d(c1, c2, KS, dilation=DIL[1])
        self.r2 = nn.ReLU()
        self.d2 = nn.Dropout(p_drop)
        self.c3 = nn.Conv1d(c2, c3, KS, dilation=DIL[2])
        self.r3 = nn.ReLU()
        self.head = nn.Conv1d(c3, n_out, 1)

    def forward(self, x):  # x: B x C x (T + RF - 1)
        x = self.d1(self.r1(self.c1(x)))
        x = self.d2(self.r2(self.c2(x)))
        x = self.r3(self.c3(x))
        return self.head(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- batching


def padded(sessions, norm):
    """Normalize and left pad each session with RF-1 zero bins (zero = train mean)."""
    out = []
    for s in sessions:
        xp = np.concatenate([np.zeros((RF - 1, s["x"].shape[1]), np.float32), norm.x(s["x"])])
        out.append((xp, s["y"] / norm.ysd, s["m"]))
    return out


def sample_batch(data, rng, bs=64, T=128):
    xs, ys, ms = [], [], []
    lens = np.array([len(d[1]) for d in data], float)
    idx = rng.choice(len(data), size=bs, p=lens / lens.sum())
    for i in idx:
        xp, y, m = data[i]
        TT = min(T, len(y))
        t0 = rng.integers(0, len(y) - TT + 1)
        xs.append(xp[t0:t0 + TT + RF - 1].T)
        ys.append(y[t0:t0 + TT].T)
        ms.append(m[t0:t0 + TT])
    TT = min(len(a[0]) for a in ys)
    xs = np.stack([a[:, :TT + RF - 1] for a in xs])
    ys = np.stack([a[:, :TT] for a in ys])
    ms = np.stack([a[:TT] for a in ms])
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ms)


def train(model, data, steps, lr=1e-3, wd=1e-4, seed=0, params=None, noise=0.0, val=None,
          eval_every=250, norm=None, bs=64, T=128, resume=None, stop_at=None):
    """Masked MSE training with AdamW + cosine schedule over `steps`.
    If `val` (raw sessions) is given, keeps the best val-R2 weights.
    `resume`/`stop_at` let a long run be split into chunks: pass a path, and training stops
    after step `stop_at` and saves its full state there; calling again with the same path
    continues from that step. Returns True when the run is finished."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state, start = -1e9, None, 1
    if resume is not None and os.path.exists(resume):
        st = torch.load(resume, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        best, best_state, start = st["best"], st["best_state"], st["step"] + 1
    end = steps if stop_at is None else min(stop_at, steps)
    for step in range(start, end + 1):
        model.train()
        x, y, m = sample_batch(data, rng, bs=bs, T=T)
        if noise:
            x = x + noise * torch.randn_like(x)
        pred = model(x)
        loss = (((pred - y) ** 2).mean(1) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if val is not None and (step % eval_every == 0 or step == steps):
            r2 = falcon_r2(model, val, norm)["mean"]
            if r2 > best:
                best = r2
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if end < steps:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        best=best, best_state=best_state, step=end), resume)
        return False
    if best_state is not None:
        model.load_state_dict(best_state)
    return True


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def predict(model, s, norm, dtype=torch.float32):
    model.eval()
    xp = np.concatenate([np.zeros((RF - 1, s["x"].shape[1]), np.float32), norm.x(s["x"])])
    x = torch.from_numpy(xp.T[None]).to(dtype)
    return model(x)[0].float().numpy().T * norm.ysd


def falcon_r2(model, sessions, norm, dtype=torch.float32):
    """Per-session variance-weighted R2 on eval-mask bins, computed with FALCON's own
    FalconEvaluator.compute_metrics_regression, then mean and std across sessions
    (same definition as FALCON's 'R2 Mean' / 'R2 Std.')."""
    per = []
    for s in sessions:
        p = predict(model, s, norm, dtype)
        r = FalconEvaluator.compute_metrics_regression(p, s["y"][s["m"]], s["m"], {s["ses"]: [len(s["m"])]})
        per.append(float(r["R2 Mean"]))
    return dict(mean=float(np.mean(per)), std=float(np.std(per)), per_session=per)


# ---------------------------------------------------------------- compression helpers


def example_input(n_in=176, T=64):
    return torch.randn(1, n_in, T + RF - 1)


def ptq_int8(fp32_model, calib_sessions, norm, n_batches=20, seed=0):
    """Static post-training INT8 quantization (FX graph mode, x86 backend), observers
    calibrated on the given sessions."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = "x86"
    m = copy_model(fp32_model).eval()
    pm = prepare_fx(m, get_default_qconfig_mapping("x86"), (example_input(),))
    data = padded(calib_sessions, norm)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _, _ = sample_batch(data, rng, bs=32, T=256)
            pm(x)
    return convert_fx(pm)


def prepare_qat(fp32_model):
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx
    torch.backends.quantized.engine = "x86"
    m = copy_model(fp32_model).train()
    return prepare_qat_fx(m, get_default_qat_qconfig_mapping("x86"), (example_input(),))


def convert_qat(qat_model):
    from torch.ao.quantization.quantize_fx import convert_fx
    import copy
    m = copy.deepcopy(qat_model).eval()
    return convert_fx(m)


def copy_model(m):
    import copy
    return copy.deepcopy(m)


def prune_channels(model, keep_frac):
    """Structured pruning: keep the top keep_frac output channels (L1 norm of filters) in each
    hidden conv layer and physically remove the rest, so the result is a genuinely smaller
    dense model."""
    c1, c2, c3 = model.c1, model.c2, model.c3
    def top(conv):
        k = max(1, int(round(conv.out_channels * keep_frac)))
        score = conv.weight.detach().abs().sum((1, 2))
        return torch.sort(torch.topk(score, k).indices).values
    i1, i2, i3 = top(c1), top(c2), top(c3)
    new = TCN(n_in=c1.in_channels, n_out=model.head.out_channels, ch=(len(i1), len(i2), len(i3)),
              p_drop=model.d1.p)
    with torch.no_grad():
        new.c1.weight.copy_(c1.weight[i1]); new.c1.bias.copy_(c1.bias[i1])
        new.c2.weight.copy_(c2.weight[i2][:, i1]); new.c2.bias.copy_(c2.bias[i2])
        new.c3.weight.copy_(c3.weight[i3][:, i2]); new.c3.bias.copy_(c3.bias[i3])
        new.head.weight.copy_(model.head.weight[:, i3]); new.head.bias.copy_(model.head.bias)
    return new


def cpu_latency_ms(model, dtype=torch.float32, n=300, threads=1):
    """Median wall time for one streaming step on CPU: one RF-length window in, one bin out,
    batch 1, single thread. A rough proxy only, NOT a Jetson number."""
    torch.set_num_threads(threads)
    x = torch.randn(1, 176, RF).to(dtype)
    model.eval()
    ts = []
    with torch.no_grad():
        for _ in range(30):
            model(x)
        for _ in range(n):
            t = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def model_bytes(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
