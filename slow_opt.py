import torch
import numpy as np
import sys
import argparse
#import matplotlib as mpl
#import matplotlib.pyplot as plt

from torch.nn.functional import softplus
from math import sqrt,sin,cos
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from fdtd_pt.opt import auto_opt
from tqdm import trange

parser = argparse.ArgumentParser(description='Optimize current absorption profile')
parser.add_argument('--cells', type=int, default=8)
parser.add_argument('--learnse', action='store_false')
parser.add_argument('--learnsb', action='store_true')
parser.add_argument('--speed', type=float,default=0.3)
parser.add_argument('--smooth',action='store_true')
parser.add_argument('--filtw', type=int,default=4)
parser.add_argument('--res', type=int,default=32)
parser.add_argument('--x0', type=float,default=-1.0)
parser.add_argument('--niter', type=int,default=10000)
parser.add_argument('--lr', type=float,default=0.01)
args = parser.parse_args(sys.argv)

n=args.cells
init = (0.,10.,10.)
theta = 0
v = args.speed
vx = v*cos(theta)
vy = v*sin(theta)
smooth=args.smooth
filter_n=args.filtw

best,hist,Bf,Ef,big_L = auto_opt(args.res,n,vx=vx,vy=vy,x0=args.x0,n_iter=args.niter,init=init,learn_se=args.learnse,learn_sb=args.learnse,lr=args.lr,smooth_current=smooth,filter_n=filter_n)
torch.save(best['alpha'],f'/mnt/PAULO/mark/pml/2D/alpha_profile_slow_{n}.pyt')
torch.save(best['sigma'],f'/mnt/PAULO/mark/pml/2D/sigma_0_slow_{n}.pyt')
torch.save(best['sigmastar'],f'/mnt/PAULO/mark/pml/2D/sigmastar_0_slow_{n}.pyt')
torch.save(hist,f'/mnt/PAULO/mark/pml/2D/hist_0_slow_{n}.pyt')
torch.save(big_L,f'/mnt/PAULO/mark/pml/2D/big_L_0_slow_{n}.pyt')