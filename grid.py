import torch

torch.set_default_dtype(torch.float32)

# define spatial tensors (x,xx,yy) on CPU
@torch.jit.script
def grid_setup(ndelta, res, Lx, Ly):
    device = ndelta.device
    nx = 2 * Lx * res + 1
    x = torch.linspace(-Lx, Lx, nx.item(), device=device)

    dx = x[1] - x[0]
    pmlx = torch.stack([Lx + n * dx for n in torch.arange(1, ndelta.item() + 1)])

    x = torch.cat((-1 * torch.flipud(pmlx), x, pmlx))
    dx = x[1] - x[0]
    nx = x.shape[0]
    ny = 2 * Ly * res + 1
    y = torch.linspace(-Ly, Ly, ny.item(), device=device)

    dy = y[1] - y[0]
    pmly = torch.stack([Ly + n * dy for n in torch.arange(1, ndelta.item() + 1)])

    y = torch.cat((-1 * torch.flipud(pmly), y, pmly))
    dy = y[1] - y[0]
    ny = y.shape[0]
    delta = ndelta * dx
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    in_sim = torch.ones_like(xx)
    in_sim[0:ndelta, :] *= 0.0
    in_sim[-ndelta:, :] *= 0.0
    in_sim[:, 0:ndelta] *= 0.0
    in_sim[:, -ndelta:] *= 0.0
    return x, y, xx, yy, delta, in_sim, dx


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
def get_sigma(se, sb, xx, yy, ndelta, Lx, Ly):
    sigmax = torch.zeros_like(xx)
    sigmastarx = torch.zeros_like(xx)
    sigmay = torch.zeros_like(xx)
    sigmastary = torch.zeros_like(xx)

    if se.dim() == 0:
        dx = xx[1, 0] - xx[0, 0]
        delta = dx * ndelta
        sigmax[0:ndelta, :] += se * torch.square((xx + Lx) / (delta))[0:ndelta, :]
        sigmay[:, 0:ndelta] += se * torch.square((yy + Ly) / (delta))[:, 0:ndelta]
        sigmax[-ndelta:, :] += se * torch.square((xx - Lx) / (delta))[-ndelta:, :]
        sigmay[:, -ndelta:] += se * torch.square((yy - Ly) / (delta))[:, -ndelta:]
        sigmastarx[0:ndelta, :] += sb * torch.square((xx + Lx) / (delta))[0:ndelta, :]
        sigmastary[:, 0:ndelta] += sb * torch.square((yy + Ly) / (delta))[:, 0:ndelta]
        sigmastarx[-ndelta:, :] += sb * torch.square((xx - Lx) / (delta))[-ndelta:, :]
        sigmastary[:, -ndelta:] += sb * torch.square((yy - Ly) / (delta))[:, -ndelta:]
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
    Lx,
    Ly,
    dt,
):
    sigmax, sigmay, sigmastarx, sigmastary = get_sigma(se, sb, xx, yy, ndelta, Lx, Ly)
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


def identity_kernel(m, ndelta, arr):
    ret = torch.zeros((arr.shape[0], arr.shape[1], m, m), device=arr.device)
    middle = int((m + 1) / 2) - 1
    ret[ndelta:-ndelta,ndelta:-ndelta, middle, middle] += 1
    return ret


# @torch.jit.script
def get_alpha(alpha0, arr):
    if type(alpha0) == tuple:
        if len(alpha0[0].shape) == 1:
            n = alpha0[0].shape[0]
            alpha = (torch.ones_like(arr), torch.ones_like(arr), torch.ones_like(arr))
            for i, a in enumerate(alpha0):
                alphax = torch.ones((arr.shape[0], 1), device=arr.device)
                alphay = torch.ones((1, arr.shape[1]), device=arr.device)
                alphax[0:n, 0] *= torch.flipud(a)
                alphax[-n:, 0] *= a
                alphay[0, 0:n] *= torch.flipud(a)
                alphay[0, -n:] *= a
                alpha[i][:] *= alphax
                alpha[i][:] *= alphay
        else:
            n = alpha0[0].shape[0]
            m = alpha0[0].shape[1]
            alpha = (
                identity_kernel(m, n, arr),
                identity_kernel(m, n, arr),
                identity_kernel(m, n, arr),
            )
            #alpha = (
            #    torch.ones((arr.shape[0], arr.shape[1], m, m), device=arr.device),
            #    torch.ones((arr.shape[0], arr.shape[1], m, m), device=arr.device),
            #    torch.ones((arr.shape[0], arr.shape[1], m, m), device=arr.device),
            #)
            for i, a in enumerate(alpha0):
                alphax = torch.zeros((arr.shape[0], 1, m, m), device=arr.device)
                alphay = torch.zeros((1, arr.shape[1], m, m), device=arr.device)
                alphax[0:n, 0, ...] += torch.flipud(a)
                alphax[-n:, 0, ...] += a
                alphay[0, 0:n, ...] += torch.flipud(a)
                alphay[0, -n:, ...] += a
                alpha[i][:] += alphax
                alpha[i][:] += alphay
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


def hole_cut(arr, i, j):
    r = torch.zeros_like(arr)
    r[i, j] += 1
    return r


def apply_alpha(alpha, J):
    if len(alpha.shape) == 2:
        return alpha * J
    pad = int((alpha.shape[-1] - 1) / 2)
    Jret = torch.zeros_like(J)
    jpos = [(p[0],p[1]) for p in torch.argwhere(J)]
    if len(jpos)==0:
        return J
    pick_stack = torch.stack([torch.unsqueeze(hole_cut(J,px,py),dim=1) for px, py in jpos])
    jconv = torch.reshape(J, (1, 1, J.shape[0], J.shape[1])).expand(1,len(jpos),-1,-1)
    alphaconv = torch.stack([alpha[px : px + 1, py : py + 1, :, :] for px,py in jpos],dim=1)
    jcout = pick_stack*torch.nn.functional.conv2d(jconv,alphaconv,padding=pad,groups=len(jpos))
    Jret += torch.sum(jcout,dim=(0,1))
    return Jret
