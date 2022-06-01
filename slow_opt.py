import torch
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt

from torch.nn.functional import softplus
from math import sqrt,sin,cos
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from fdtd_pt.opt import auto_opt
from tqdm import trange


n=8
init = (0.,10.,10.)
theta = 0
v = 0.3
vx = v*cos(theta)
vy = v*sin(theta)
smooth=True
filter_n=4

best,hist,Bf,Ef,big_L = auto_opt(32,n,vx=vx,vy=vy,x0=-1.0,n_iter=1000,init=init,learn_se=True,learn_sb=False,lr=0.001,smooth_current=smooth,filter_n=filter_n)
torch.save(best['alpha'],f'/mnt/PAULO/mark/pml/2D/alpha_profile_slow_{n}.pyt')
torch.save(best['sigma'],f'/mnt/PAULO/mark/pml/2D/sigma_0_slow_{n}.pyt')
torch.save(best['sigmastar'],f'/mnt/PAULO/mark/pml/2D/sigmastar_0_slow_{n}.pyt')
torch.save(hist,f'/mnt/PAULO/mark/pml/2D/hist_0_slow_{n}.pyt')
torch.save(big_L,f'/mnt/PAULO/mark/pml/2D/big_L_0_slow_{n}.pyt')