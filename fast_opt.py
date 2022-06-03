import torch
import numpy as np
import sys
import argparse

# import matplotlib as mpl
# import matplotlib.pyplot as plt

from torch.nn.functional import softplus
from math import sqrt, sin, cos
import os
import sys

sys.path.insert(0, os.path.abspath("../"))
from fdtd_pt.opt import auto_opt, func
from tqdm import trange

parser = argparse.ArgumentParser(description="Optimize current absorption profile")
parser.add_argument("--cells", type=int, default=8)
parser.add_argument("--learnse", action="store_false")
parser.add_argument("--learnsb", action="store_true")
parser.add_argument("--gamma", type=float, default=10)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--filtw", type=int, default=4)
parser.add_argument("--res", type=int, default=32)
parser.add_argument("--x0", type=float, default=-1.0)
parser.add_argument("--niter", type=int, default=10000)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()


smooth = args.smooth
filter_n = args.filtw

strmod = "fast"
if smooth:
    strmod += f"_smooth_{filter_n}"
if args.learnsb:
    strmod += "_sb"

n = args.cells
try:
    alpha = func(torch.load(f"/mnt/PAULO/mark/pml/2D/alpha_profile_{strmod}_{n}.pyt"))
    sigma = softplus(torch.load(f"/mnt/PAULO/mark/pml/2D/sigma_0_{strmod}_{n}.pyt"))
    sigmastar = softplus(
        torch.load(f"/mnt/PAULO/mark/pml/2D/sigmastar_0_{strmod}_{n}.pyt")
    )
    init = (alpha.numpy(), float(sigma), float(sigmastar))
except FileNotFoundError:
    init = (0.0, 10.0, 10.0)
theta = 0
v = sqrt(args.gamma**2 - 1) / args.gamma
vx = v * cos(theta)
vy = v * sin(theta)

best, hist, Bf, Ef, big_L = auto_opt(
    args.res,
    n,
    vx=vx,
    vy=vy,
    x0=args.x0,
    n_iter=args.niter,
    init=init,
    learn_se=args.learnse,
    learn_sb=args.learnse,
    lr=args.lr,
    smooth_current=smooth,
    filter_n=filter_n,
)

torch.save(best["alpha"], f"/mnt/PAULO/mark/pml/2D/alpha_profile_{strmod}_{n}.pyt")
torch.save(best["sigma"], f"/mnt/PAULO/mark/pml/2D/sigma_0_{strmod}_{n}.pyt")
torch.save(best["sigmastar"], f"/mnt/PAULO/mark/pml/2D/sigmastar_0_{strmod}_{n}.pyt")
torch.save(hist, f"/mnt/PAULO/mark/pml/2D/hist_0_{strmod}_{n}.pyt")
torch.save(big_L, f"/mnt/PAULO/mark/pml/2D/big_L_0_{strmod}_{n}.pyt")
