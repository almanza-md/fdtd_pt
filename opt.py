import torch
import numpy as np
from torch.nn.functional import softplus
from .grid import grid_setup
from .sim import sim_setup, sim, sim_bigbox
from tqdm import trange
from math import sqrt

torch.set_default_dtype(torch.float32)
alph0 = torch.atanh(torch.tensor(0.9999))


@torch.jit.script
def func(x):
    return (1 + torch.tanh(x)) / 2


def auto_opt(
    resolution,
    ndelta,
    x0=0.0,
    y0=0.0,
    vx=0.0,
    vy=0.0,
    init=(0.0, 4 / 0.0315, 4 / 0.0315),
    n_iter=300,
    loop=False,
    lr=0.1,
    learn_se=False,
    learn_sb=False,
    smooth_current=False,filter_n=1,vec_a=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if vec_a:
        if type(init[0]) == float:
            if init[0] == 0:
                ax = torch.linspace(
                    start=alph0,
                    end=-alph0,
                    steps=ndelta,
                    requires_grad=True,
                    dtype=torch.float32,
                    device=device,
                )
                ay = torch.linspace(
                    start=alph0,
                    end=-alph0,
                    steps=ndelta,
                    requires_grad=True,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                ax = init[0] * torch.ones(
                    ndelta,
                    dtype=torch.float32,
                    device=device,
                )
                ax.requires_grad = True
                ay = init[0] * torch.ones(
                    ndelta,
                    dtype=torch.float32,
                    device=device,
                )
                ay.requires_grad = True
        else:
            ax = torch.tensor(init[0],dtype=torch.float32,requires_grad=True,device=device)
            ay = torch.tensor(init[0],dtype=torch.float32,requires_grad=True,device=device)
    else:
        if type(init[0]) == float:
            if init[0] == 0:
                a = torch.linspace(
                    start=alph0,
                    end=-alph0,
                    steps=ndelta,
                    requires_grad=True,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                a = init[0] * torch.ones(
                    ndelta,
                    dtype=torch.float32,
                    device=device,
                )
                a.requires_grad = True
        else:
            a = torch.tensor(init[0],dtype=torch.float32,requires_grad=True,device=device)
    se = torch.tensor(
        init[1], dtype=torch.float32, requires_grad=learn_se, device=device
    )
    if vec_a:
        params = [ax,ay]
    else:
        params = [a]
    if learn_se:
        params.append(se)
    if learn_sb:
        sb = torch.tensor(
            init[2], dtype=torch.float32, requires_grad=learn_sb, device=device
        )
        params.append(sb)
    else:
        sb = se
    a_opt = torch.optim.NAdam(params, lr=lr)

    #lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(a_opt,factor=0.5,threshold=1e-12, threshold_mode='abs')

    loss = 0.0

    loss_hist = []
    a_hist = []
    se_hist = []
    sb_hist = []
    a_best = 0.0
    se_best = 0.0
    sb_best = 0.0
    resolution = torch.tensor(resolution, requires_grad=False, device=device)
    ndelta = torch.tensor(ndelta, requires_grad=False, device=device)
    x0 = torch.tensor(x0, requires_grad=False)
    y0 = torch.tensor(y0, requires_grad=False)
    #vx = torch.tensor(vx, requires_grad=False)
    #vy = torch.tensor(vy, requires_grad=False)
    (
        x,
        t,
        xx,
        yy,
        delta,
        in_sim,
        Jloader,
        dx,
        dt,
        maskb,
        maskex,
        maskey,
        maskez,
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
    ) = sim_setup(
        ndelta=ndelta,
        res=resolution,
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        L=torch.tensor(2, device=device), smooth=smooth_current,filter_n=filter_n
    )
    big_L = round(float(t[-1]))#*sqrt((vx)**2 + (vy)**2))
    (
        xb,
        tb,
        xxb,
        yyb,
        deltab,
        in_simb,
        Jloaderb,
        dxb,
        dtb,
        maskbb,
        maskexb,
        maskeyb,
        maskezb,
        e_xb,
        e_yb,
        e_zxb,
        e_zyb,
        b_xb,
        b_yb,
        b_zxb,
        b_zyb,
    ) = sim_setup(
        ndelta=ndelta,
        res=resolution,
        x0=x0,
        y0=y0,
        vx=vx,
        vy=vy,
        L=torch.tensor(big_L, device=device), smooth=smooth_current,filter_n=filter_n
    )
    Bf, Ef, xx_big, *_ = sim_bigbox(
        se.detach(),
        sb.detach(),
        tb,
        xxb,
        yyb,
        ndelta,
        torch.tensor(big_L),
        Jloaderb,
        dxb,
        dtb,
        maskbb,
        maskexb,
        maskeyb,
        maskezb,
        e_xb,
        e_yb,
        e_zxb,
        e_zyb,
        b_xb,
        b_yb,
        b_zxb,
        b_zyb,
    )
    big0 = torch.argmin(torch.abs(xx_big[:, 0]))
    small0 = torch.argmin(torch.abs(xx[:, 0]))
    Bf = Bf[
        big0 - small0 : big0 + small0 + 1, big0 - small0 : big0 + small0 + 1, :
    ].clone()
    Ef = Ef[
        big0 - small0 : big0 + small0 + 1, big0 - small0 : big0 + small0 + 1, :
    ].clone()
    Uref = torch.sum(torch.square(Ef) + torch.square(Bf))
    
    for i in trange(n_iter):
        a_opt.zero_grad()
        if vec_a:
            func_a = (func(ax),func(ay))
        else:
            func_a = func(a)
        loss = sim(
            func_a,
            softplus(se),
            softplus(sb),
            xx,
            yy,
            ndelta,
            torch.tensor(2, device=device),
            in_sim,
            Jloader,
            dx,
            dt,
            maskb,
            maskex,
            maskey,
            maskez,
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            Ef,
            Bf,
        )
        loss /= Uref
        loss.backward()
        l = loss.detach()
        if vec_a:
            if i == 0 or l < min(loss_hist):
                a_best = (ax.detach().cpu().clone(),ay.detach().cpu().clone())
                se_best = se.detach().cpu().clone()
                sb_best = sb.detach().cpu().clone()
            a_hist.append((ax.detach().cpu().clone(),ay.detach().cpu().clone()))
        else:
            if i == 0 or l < min(loss_hist):
                a_best = a.detach().cpu().clone()
                se_best = se.detach().cpu().clone()
                sb_best = sb.detach().cpu().clone()
            a_hist.append(a.detach().cpu().clone())
        se_hist.append(se.detach().cpu().clone())
        sb_hist.append(sb.detach().cpu().clone())
        loss_hist.append(l)

        a_opt.step()
        #lr_sched.step(loss)
    if loop:
        while not np.isclose(
            np.log(np.mean(loss_hist[-25:])) - np.log(np.mean(loss_hist[-100:-75])), 0.0
        ):
            a_opt.zero_grad()
            loss = sim(
                func(a),
                se,
                se,
                xx,
                yy,
                ndelta,
                torch.tensor(2, device=device),
                in_sim,
                Jloader,
                dx,
                dt,
                maskb,
                maskex,
                maskey,
                maskez,
                Ef,
                Bf,
            )
            loss.backward()
            l = loss.detach().item()
            if i == 0 or l < min(loss_hist):
                a_best = (a.clone().detach().cpu(), se.clone().detach().item())
            a_hist.append((a.clone().detach().cpu(), se.clone().detach().item()))
            loss_hist.append(l)

            a_opt.step()
    return (
        {"alpha": a_best, "sigma": se_best, "sigmastar": sb_best},
        {"alpha": a_hist, "sigma": se_hist, "sigmastar": sb_hist, "loss": loss_hist},
        Bf.cpu(),
        Ef.cpu(),
        big_L
    )
