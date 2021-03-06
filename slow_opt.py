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
from fdtd_pt.opt import auto_opt,func
from tqdm import trange

parser = argparse.ArgumentParser(description="Optimize current absorption profile")
parser.add_argument("--cells", type=int, default=8)
parser.add_argument("--learnse", action="store_false")
parser.add_argument("--learnsb", action="store_true")
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--filtw", type=int, default=4)
parser.add_argument("--res", type=int, default=32)
parser.add_argument("--x0", type=float, default=-1.0)
parser.add_argument("--niter", type=int, default=10000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--savedir", default='/mnt/PAULO/mark/pml/2D')
parser.add_argument("--ignoreold", action='store_true')
args = parser.parse_args()
ignoreold = args.ignoreold
save_dir = args.savedir
smooth = args.smooth
filter_n = args.filtw

strmod = "slow"
if smooth:
    strmod += f"_smooth_{filter_n}"
if args.learnsb:
    strmod += "_sb"

n = args.cells
try:
    alpha = torch.load(f"{save_dir}/alpha_profile_{strmod}_{n}.pyt")
    sigma = torch.load(f"{save_dir}/sigma_0_{strmod}_{n}.pyt")
    sigmastar = torch.load(f"{save_dir}/sigmastar_0_{strmod}_{n}.pyt")
    print('Found previous profiles')
    init = (alpha.numpy(), float(sigma), float(sigmastar))
except FileNotFoundError:
    print('No previous profiles')
    init = (0.0, 10.0, 10.0)
if ignoreold:
    print('Ignoring previous profiles')
    init = (0.0, 10.0, 10.0)
theta = 0
v = args.speed
vx = v * cos(theta)
vy = v * sin(theta)
smooth = args.smooth
filter_n = args.filtw

best, hist, Bf, Ef, big_L = auto_opt(
    args.res,
    n,
    vx=vx,
    vy=vy,
    x0=args.x0,
    n_iter=args.niter,
    init=init,
    learn_se=args.learnse,
    learn_sb=args.learnsb,
    lr=args.lr,
    smooth_current=smooth,
    filter_n=filter_n,
    checkpoint=50,
    save_dir=save_dir,strmod=strmod,
)
strmod = "slow"
if smooth:
    strmod += f"_smooth_{filter_n}"
if args.learnsb:
    strmod += "_sb"
torch.save(best["alpha"], f"{save_dir}/alpha_profile_{strmod}_{n}.pyt")
torch.save(best["sigma"], f"{save_dir}/sigma_0_{strmod}_{n}.pyt")
torch.save(best["sigmastar"], f"{save_dir}/sigmastar_0_{strmod}_{n}.pyt")
torch.save(hist, f"{save_dir}/hist_0_{strmod}_{n}.pyt")
torch.save(big_L, f"{save_dir}/big_L_0_{strmod}_{n}.pyt")

