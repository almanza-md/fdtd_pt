from typing import Tuple
import torch

torch.set_default_dtype(torch.float32)

# define spatial tensors (x,xx,yy) on CPU
@torch.jit.script
def grid_setup(ndelta, res, L):
    device = ndelta.device
    nx = 2 * L * res + 1
    x = torch.linspace(-L, L, nx.item(), device=device)

    dx = x[1] - x[0]
    pmlx = torch.stack([L + n * dx for n in torch.arange(1, ndelta.item() + 1)])

    x = torch.cat((-1 * torch.flipud(pmlx), x, pmlx))
    dx = x[1] - x[0]
    nx = x.shape[0]
    ny = nx
    y = x
    delta = ndelta * dx
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    in_sim = torch.ones_like(xx)
    in_sim[0:ndelta, :] *= 0.0
    in_sim[-ndelta:, :] *= 0.0
    in_sim[:, 0:ndelta] *= 0.0
    in_sim[:, -ndelta:] *= 0.0
    return x, xx, yy, delta, in_sim, dx


def get_sigma(
    se,
    sb,
    xx,
    yy,
    ndelta,
    L,
    dt,
):
    device = ndelta.device
    sigmax = torch.zeros_like(xx)
    sigmastarx = torch.zeros_like(xx)
    sigmay = torch.zeros_like(xx)
    sigmastary = torch.zeros_like(xx)
    dx = xx[1, 0] - xx[0, 0]
    sigmax[xx < -L] += se * torch.square((xx + L) / (6 * dx))[xx < -L]
    sigmay[yy < -L] += se * torch.square((yy + L) / (6 * dx))[yy < -L]
    sigmax[xx > L] += se * torch.square((xx - L) / (6 * dx))[xx > L]
    sigmay[yy > L] += se * torch.square((yy - L) / (6 * dx))[yy > L]
    sigmastarx[xx < -L] += sb * torch.square((xx + L) / (6 * dx))[xx < -L]
    sigmastary[yy < -L] += sb * torch.square((yy + L) / (6 * dx))[yy < -L]
    sigmastarx[xx > L] += sb * torch.square((xx - L) / (6 * dx))[xx > L]
    sigmastary[yy > L] += sb * torch.square((yy - L) / (6 * dx))[yy > L]
    return sigmax, sigmay, sigmastarx, sigmastary


@torch.jit.script
def get_CD(
    se,
    sb,
    xx,
    yy,
    ndelta,
    L,
    dt,
):
    device = ndelta.device
    sigmax = torch.zeros_like(xx)
    sigmastarx = torch.zeros_like(xx)
    sigmay = torch.zeros_like(xx)
    sigmastary = torch.zeros_like(xx)
    dx = xx[1, 0] - xx[0, 0]
    sigmax[xx < -L] += se * torch.square((xx + L) / (6 * dx))[xx < -L]
    sigmay[yy < -L] += se * torch.square((yy + L) / (6 * dx))[yy < -L]
    sigmax[xx > L] += se * torch.square((xx - L) / (6 * dx))[xx > L]
    sigmay[yy > L] += se * torch.square((yy - L) / (6 * dx))[yy > L]
    sigmastarx[xx < -L] += sb * torch.square((xx + L) / (6 * dx))[xx < -L]
    sigmastary[yy < -L] += sb * torch.square((yy + L) / (6 * dx))[yy < -L]
    sigmastarx[xx > L] += sb * torch.square((xx - L) / (6 * dx))[xx > L]
    sigmastary[yy > L] += sb * torch.square((yy - L) / (6 * dx))[yy > L]
    Dbx = ((dt / 2) / (1 + dt * sigmastarx / 4)) / dx
    Dax = (1 - dt * sigmastarx / 4) / (1 + dt * sigmastarx / 4)
    Cbx = (dt / (1 + dt * sigmax / 2)) / dx
    Cax = (1 - dt * sigmax / 2) / (1 + dt * sigmax / 2)
    Dby = ((dt / 2) / (1 + dt * sigmastary / 4)) / dx
    Day = (1 - dt * sigmastary / 4) / (1 + dt * sigmastary / 4)
    Cby = (dt / (1 + dt * sigmay / 2)) / dx
    Cay = (1 - dt * sigmay / 2) / (1 + dt * sigmay / 2)
    return Dbx, Dax, Cbx, Cax, Dby, Day, Cby, Cay


def get_alpha(alpha0, arr):
    n = alpha0.shape[0]
    alpha = torch.ones_like(arr)
    alphax = torch.ones((arr.shape[0], 1), device=arr.device)
    alphay = torch.ones((1, arr.shape[1]), device=arr.device)
    alphax[0:n, 0] *= torch.flipud(alpha0)
    alphax[-n:, 0] *= alpha0
    alphay[0, 0:n] *= torch.flipud(alpha0)
    alphay[0, -n:] *= alpha0
    alpha *= alphax
    alpha[n:-n] *= alphay
    return alpha
