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


@torch.jit.script
def sig_helper(s0, arr):
    n = s0.shape[0]
    sigma = torch.zeros_like(arr)
    sigmax = torch.zeros((arr.shape[0], 1), device=arr.device)
    sigmay = torch.zeros((1, arr.shape[1]), device=arr.device)
    sigmax[0:n, 0] += torch.flipud(s0)
    sigmax[-n:, 0] += s0
    sigmay[0, 0:n] += torch.flipud(s0)
    sigmay[0, -n:] += s0
    return sigma + sigmax, sigma + sigmay


@torch.jit.script
def get_sigma(
    se,
    sb,
    xx,
    yy,
    ndelta,
    L,
):
    sigmax = torch.zeros_like(xx)
    sigmastarx = torch.zeros_like(xx)
    sigmay = torch.zeros_like(xx)
    sigmastary = torch.zeros_like(xx)
    if se.dim() == 0:
        dx = xx[1, 0] - xx[0, 0]
        sigmax[0:ndelta, :] += se * torch.square((xx + L) / (6 * dx))[0:ndelta, :]
        sigmay[:, 0:ndelta] += se * torch.square((yy + L) / (6 * dx))[:, 0:ndelta]
        sigmax[-ndelta:, :] += se * torch.square((xx - L) / (6 * dx))[-ndelta:, :]
        sigmay[:, -ndelta:] += se * torch.square((yy - L) / (6 * dx))[:, -ndelta:]
        sigmastarx[0:ndelta, :] += sb * torch.square((xx + L) / (6 * dx))[0:ndelta, :]
        sigmastary[:, 0:ndelta] += sb * torch.square((yy + L) / (6 * dx))[:, 0:ndelta]
        sigmastarx[-ndelta:, :] += sb * torch.square((xx - L) / (6 * dx))[-ndelta:, :]
        sigmastary[:, -ndelta:] += sb * torch.square((yy - L) / (6 * dx))[:, -ndelta:]
    else:
        sigmax, sigmay = sig_helper(se, xx)
        sigmastarx, sigmastary = sig_helper(sb, xx)
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
    sigmax, sigmay, sigmastarx, sigmastary = get_sigma(se, sb, xx, yy, ndelta, L)
    dx = xx[1, 0] - xx[0, 0]
    Dbx = ((dt / 2) / (1 + dt * sigmastarx / 4)) / dx
    Dax = (1 - dt * sigmastarx / 4) / (1 + dt * sigmastarx / 4)
    Cbx = (dt / (1 + dt * sigmax / 2)) / dx
    Cax = (1 - dt * sigmax / 2) / (1 + dt * sigmax / 2)
    Dby = ((dt / 2) / (1 + dt * sigmastary / 4)) / dx
    Day = (1 - dt * sigmastary / 4) / (1 + dt * sigmastary / 4)
    Cby = (dt / (1 + dt * sigmay / 2)) / dx
    Cay = (1 - dt * sigmay / 2) / (1 + dt * sigmay / 2)
    return Dbx, Dax, Cbx, Cax, Dby, Day, Cby, Cay


@torch.jit.script
def get_alpha(alpha0, arr):
    if type(alpha0)==tuple:
        n = alpha0[0].shape[0]
        alpha = (torch.ones_like(arr),torch.ones_like(arr),torch.ones_like(arr))
        for i,a in enumerate(alpha0):
            alphax = torch.ones((arr.shape[0], 1), device=arr.device)
            alphay = torch.ones((1, arr.shape[1]), device=arr.device)
            alphax[0:n, 0] *= torch.flipud(a)
            alphax[-n:, 0] *= a
            alphay[0, 0:n] *= torch.flipud(a)
            alphay[0, -n:] *= a
            alpha[i] *= alphax
            alpha[i][n:-n] *= alphay
    else:
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
