import torch
from grid import grid_setup
from fields import masks, advance_flds
from current import jfunc

torch.set_default_dtype(torch.float32)

@torch.jit.script
def sim_setup(
    alpha0=torch.tensor(5.0),
    ndelta=torch.tensor(8),
    res=torch.tensor(32),
    se=torch.tensor(5.0),
    sb=torch.tensor(5.0),
    vx=torch.tensor(0.5),
    vy=torch.tensor(0.0),
    x0=torch.tensor(0.0),
    y0=torch.tensor(0.0),
    L=torch.tensor(2),
):
    with torch.no_grad():
        x, xx, yy, delta, e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = grid_setup(
            ndelta, res
        )
        device = x.device
        dx = x[1] - x[0]
        nx = x.shape[0]
        ny = x.shape[0]
        J_x, J_y, t = jfunc(
            x, vx, vy, L.to(torch.float32).to(device), x0=x0, y0=y0, delta=delta
        )
        dt = t[1] - t[0]
        J_z = torch.zeros_like(J_x[:, :, 0:1])

        in_sim = torch.ones_like(e_x)
        in_sim[0:ndelta, :] *= 0.0
        in_sim[-ndelta:, :] *= 0.0
        in_sim[:, 0:ndelta] *= 0.0
        in_sim[:, -ndelta:] *= 0.0

        maskb, maskex, maskey, maskez = masks(e_x)
    sigmax = torch.zeros_like(e_y)
    sigmastarx = torch.zeros_like(e_y)
    sigmay = torch.zeros_like(e_y)
    sigmastary = torch.zeros_like(e_y)

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
    alpha = torch.ones_like(e_y)
    alphax = torch.ones((nx, 1))
    alphay = torch.ones((1, ny))
    alphax[0:ndelta, 0] *= torch.flipud(alpha0)
    alphax[-ndelta:, 0] *= alpha0
    alphay[0, 0:ndelta] *= torch.flipud(alpha0)
    alphay[0, -ndelta:] *= alpha0
    alpha *= alphax
    alpha[ndelta:-ndelta] *= alphay

    return (
        x,
        t,
        xx,
        yy,
        in_sim,
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
        alpha,
        J_x,
        J_y,
        J_z,
        dx,
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
        maskb,
        maskex,
        maskey,
        maskez,
    )


@torch.jit.script
def sim(
    alpha0=torch.tensor(5.0),
    ndelta=torch.tensor(8),
    res=torch.tensor(32),
    se=torch.tensor(5.0),
    sb=torch.tensor(5.0),
    vx=torch.tensor(0.5),
    vy=torch.tensor(0.0),
    x0=torch.tensor(0.0),
    y0=torch.tensor(0.0),
    Ef=torch.tensor(0.0),
    Bf=torch.tensor(0.0),
):
    (
        x,
        t,
        xx,
        yy,
        in_sim,
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
        alpha,
        J_x,
        J_y,
        J_z,
        dx,
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
        maskb,
        maskex,
        maskey,
        maskez,
    ) = sim_setup(alpha0, ndelta, res, se, sb, vx, vy, x0, y0)
    for i in torch.arange(0, t.shape[0]):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x[:, :, i] * alpha,
            J_y[:, :, i] * alpha,
            J_z[:, :, 0],
            dx,
            Cax,
            Cbx,
            Dax,
            Dbx,
            Cay,
            Cby,
            Day,
            Dby,
            maskb,
            maskex,
            maskey,
            maskez,
        )

    u = (
        torch.square(e_x - Ef[:, :, 0])
        + torch.square(e_y - Ef[:, :, 1])
        + torch.square(e_zx + e_zy - Ef[:, :, 2])
        + torch.square(b_x - Bf[:, :, 0])
        + torch.square(b_y - Bf[:, :, 1])
        + torch.square(b_zx + b_zy - Bf[:, :, 2])
    )
    u *= in_sim
    Utot = torch.sum(u)
    return Utot


def sim_EB(
    ndelta,
    res=torch.tensor(32),
    se=torch.tensor(5.0),
    sb=torch.tensor(5.0),
    vx=torch.tensor(0.5),
    vy=torch.tensor(0.0),
    alpha0=torch.tensor(5.0),
    x0=torch.tensor(0.0),
    y0=torch.tensor(0.0),
):
    (
        x,
        t,
        xx,
        yy,
        in_sim,
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
        alpha,
        J_x,
        J_y,
        J_z,
        dx,
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
        maskb,
        maskex,
        maskey,
        maskez,
    ) = sim_setup(alpha0, ndelta, res, se, sb, vx, vy, x0, y0)
    nx = x.shape[0]
    Barr = torch.zeros((nx, nx, 3, t.shape[0]))
    Earr = torch.zeros((nx, nx, 3, t.shape[0]))
    for i in torch.arange(0, t.shape[0]):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x[..., i] * alpha,
            J_y[..., i] * alpha,
            J_z[..., 0],
            dx,
            Cax,
            Cbx,
            Dax,
            Dbx,
            Cay,
            Cby,
            Day,
            Dby,
            maskb,
            maskex,
            maskey,
            maskez,
        )
        Barr[..., 0, i] = b_x
        Barr[..., 1, i] = b_y
        Barr[..., 2, i] = b_zx + b_zy
        Earr[..., 0, i] = e_x
        Earr[..., 1, i] = e_y
        Earr[..., 2, i] = e_zx + e_zy
    return Barr, Earr, xx, yy, t


def sim_bigbox(
    ndelta,
    res=torch.tensor(32),
    se=torch.tensor(5.0),
    sb=torch.tensor(5.0),
    vx=torch.tensor(0.5),
    vy=torch.tensor(0.0),
    alpha0=torch.tensor(5.0),
    x0=torch.tensor(0.0),
    y0=torch.tensor(0.0),
):
    (
        x,
        t,
        xx,
        yy,
        in_sim,
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
        alpha,
        J_x,
        J_y,
        J_z,
        dx,
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
        maskb,
        maskex,
        maskey,
        maskez,
    ) = sim_setup(alpha0, ndelta, res, se, sb, vx, vy, x0, y0, L=torch.tensor(8))
    nx = x.shape[0]
    Barr = torch.zeros((nx, nx, 3))
    Earr = torch.zeros((nx, nx, 3))

    for i in range(t.shape[0]):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x[..., i],
            J_y[..., i],
            J_z[..., 0],
            dx,
            Cax,
            Cbx,
            Dax,
            Dbx,
            Cay,
            Cby,
            Day,
            Dby,
            maskb,
            maskex,
            maskey,
            maskez,
        )
    Barr[..., 0] = b_x
    Barr[..., 1] = b_y
    Barr[..., 2] = b_zx + b_zy
    Earr[..., 0] = e_x
    Earr[..., 1] = e_y
    Earr[..., 2] = e_zx + e_zy

    return Barr, Earr, xx, yy, t
