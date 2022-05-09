import torch

torch.set_default_dtype(torch.float32)


@torch.jit.script
def grid_setup(ndelta, res, L):
    device = ndelta.device
    nx = 2 * L * res.cpu() + 1  # torch.floor(2*L*res).to(torch.int64)
    x = torch.linspace(-L, L, nx.item())

    dx = x[1] - x[0]
    pmlx = torch.stack([L + n * dx for n in torch.arange(1, ndelta.item() + 1)])

    x = torch.cat((-1 * torch.flipud(pmlx), x, pmlx))
    # x = x.to(device=device)
    dx = x[1] - x[0]
    # dx = dx.to(device)
    nx = x.shape[0]
    ny = nx
    y = x
    delta = ndelta * dx
    e_x = torch.zeros((nx, ny), device=device)
    e_y = torch.zeros((nx, ny), device=device)
    e_zx = torch.zeros((nx, ny), device=device)
    e_zy = torch.zeros((nx, ny), device=device)
    b_x = torch.zeros((nx, ny), device=device)
    b_y = torch.zeros((nx, ny), device=device)
    b_zx = torch.zeros((nx, ny), device=device)
    b_zy = torch.zeros((nx, ny), device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return x, xx, yy, delta, e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy


def alpha_sigma(se, sb, alpha0, xx, yy, ndelta, L):
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
    alpha = torch.ones_like(xx)
    alphax = torch.ones((xx.shape[0], 1))
    alphay = torch.ones((1, xx.shape[1]))
    alphax[0:ndelta, 0] *= torch.flipud(alpha0)
    alphax[-ndelta:, 0] *= alpha0
    alphay[0, 0:ndelta] *= torch.flipud(alpha0)
    alphay[0, -ndelta:] *= alpha0
    alpha *= alphax
    alpha[ndelta:-ndelta] *= alphay
    return alpha, sigmax, sigmay, sigmastarx, sigmastary
