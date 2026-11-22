"""Bin FALCON H1 NWB files into per-session npz using falcon_challenge's own loader."""
import sys, glob, os, re
import numpy as np
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.config import FalconTask

ROOT = os.environ.get("FALCON_DIR", "/workspace/work/data/falcon/000954")
OUT = os.environ.get("FALCON_NPZ", "/workspace/work/data/falcon/h1_npz")
os.makedirs(OUT, exist_ok=True)
for split in ["held-in-calib", "held-in-minival", "held-out-calib"]:
    for fn in sorted(glob.glob(f"{ROOT}/sub-HumanPitt-{split}/*.nwb")):
        ses = re.search(r"ses-(\w+)\.nwb", fn).group(1)
        x, y, tc, m = load_nwb(fn, dataset=FalconTask.h1)
        np.savez(f"{OUT}/{split}_{ses}.npz", x=x, y=y, trial_change=tc, eval_mask=m)
        print(split, ses, x.shape, y.shape, m.mean().round(3))
