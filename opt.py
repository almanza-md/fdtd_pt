import torch
import numpy as np
from torch.nn.functional import softplus
from .grid import grid_setup
from .sim import sim, sim_bigbox
from tqdm import trange

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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        a = torch.tensor(
            init[0],
            requires_grad=True,
            dtype=torch.float32,
            device=device,
        )
    se = torch.tensor(init[1], dtype=torch.float32, requires_grad=True, device=device)
    a_opt = torch.optim.Adam((a, se), lr=0.4)
    loss = 0.0
    if not loop:
        loss_hist = torch.zeros(n_iter, device=device)
        a_hist = torch.zeros((n_iter, a.shape[0]), device=device)
        se_hist = torch.zeros(n_iter, device=device)
    else:
        loss_hist = []
        a_hist = []
        se_hist = []
    a_best = 0.0
    se_best = 0.0
    resolution = torch.tensor(resolution, requires_grad=False, device=device)
    ndelta = torch.tensor(ndelta, requires_grad=False, device=device)
    x0 = torch.tensor(x0, requires_grad=False)
    y0 = torch.tensor(y0, requires_grad=False)
    vx = torch.tensor(vx, requires_grad=False)
    vy = torch.tensor(vy, requires_grad=False)

    Bf, Ef, xx, *_ = sim_bigbox(
        ndelta,
        res=resolution,
        se=torch.tensor(0.0, device=device),
        sb=torch.tensor(0.0, device=device),
        vx=vx,
        vy=vy,
        alpha0=a,
        x0=x0,
        y0=y0,
    )
    _, xxs, *_ = grid_setup(ndelta, res=resolution, L=torch.tensor(2))
    big0 = torch.argmin(torch.abs(xx[:, 0]))
    small0 = torch.argmin(torch.abs(xxs[:, 0]))
    Bf = Bf[big0 - small0 : big0 + small0 + 1, big0 - small0 : big0 + small0 + 1, :].to(
        device
    )
    Ef = Ef[big0 - small0 : big0 + small0 + 1, big0 - small0 : big0 + small0 + 1, :].to(
        device
    )
    for i in trange(n_iter):
        a_opt.zero_grad()
        loss = sim(
            alpha0=func(a),
            ndelta=ndelta,
            res=resolution,
            se=softplus(se),
            sb=softplus(se),
            x0=x0,
            y0=y0,
            vx=vx,
            vy=vy,
            Ef=Ef,
            Bf=Bf,
            L=torch.tensor(2),
        )
        loss.backward()
        l = loss.detach()
        if i == 0 or l < torch.min(loss_hist):
            a_best = a.clone().detach()
            se_best = se.clone().detach()
        a_hist[i, :] = a.clone().detach()
        se_hist[i] = se.clone().detach()
        loss_hist[i] = l

        a_opt.step()
    if loop:
        while not np.isclose(
            np.log(np.mean(loss_hist[-25:])) - np.log(np.mean(loss_hist[-100:-75])), 0.0
        ):
            a_opt.zero_grad()
            loss = sim(
                alpha0=func(a),
                ndelta=ndelta,
                res=resolution,
                se=softplus(se),
                sb=softplus(se),
                x0=x0,
                y0=y0,
                vx=vx,
                vy=vy,
                Ef=Ef,
                Bf=Bf,
                L=torch.tensor(2),
            )
            loss.backward()
            l = loss.detach().item()
            if i == 0 or l < min(loss_hist):
                a_best = (a.clone().detach().cpu(), se.clone().detach().item())
            a_hist.append((a.clone().detach().cpu(), se.clone().detach().item()))
            loss_hist.append(l)

            a_opt.step()
    return a_best.cpu(),se_best.cpu(), a_hist.cpu(),se_hist.cpu(), loss_hist.cpu(), Bf.cpu(), Ef.cpu()
