import torch
from .grid import grid_setup, get_alpha, get_sigma
from .fields import masks, advance_flds
from .current import jfunc

torch.set_default_dtype(torch.float32)


#@torch.jit.script
def sim_setup(
    alpha0,
    ndelta,
    res,
    se,
    sb,
    vx,
    vy,
    x0,
    y0,
    L,
):
    with torch.no_grad():
        x, xx, yy, delta, e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = grid_setup(
            ndelta, res, L
        )
        device = e_x.device
        dx = x[1] - x[0]
        # nx = x.shape[0]
        # ny = x.shape[0]
        J_x, J_y, t = jfunc(x, vx, vy, L.to(torch.float32), x0=x0, y0=y0, delta=delta)
        dt = t[1] - t[0]
        J_z = torch.zeros_like(J_x)

        J = torch.utils.data.TensorDataset(J_x,J_y,J_z)
        Jloader = torch.utils.data.DataLoader(J,num_workers=2,pin_memory=True)

        in_sim = torch.ones_like(e_x)
        in_sim[0:ndelta, :] *= 0.0
        in_sim[-ndelta:, :] *= 0.0
        in_sim[:, 0:ndelta] *= 0.0
        in_sim[:, -ndelta:] *= 0.0

        maskb, maskex, maskey, maskez = masks(e_x)
    sigmax, sigmay, sigmastarx, sigmastary = get_sigma(
        se, sb, xx, yy, ndelta, L
    )
    alpha = get_alpha(alpha0, e_x)
    dt = dt.to(device)
    dx = dx.to(device)
    Dbx = ((dt / 2) / (1 + dt * sigmastarx / 4)) / dx
    #Dbx = Dbx.to(device)
    Dax = (1 - dt * sigmastarx / 4) / (1 + dt * sigmastarx / 4)
    #Dax = Dax.to(device)
    Cbx = (dt / (1 + dt * sigmax / 2)) / dx
    #Cbx = Cbx.to(device)
    Cax = (1 - dt * sigmax / 2) / (1 + dt * sigmax / 2)
    #Cax = Cax.to(device)
    Dby = ((dt / 2) / (1 + dt * sigmastary / 4)) / dx
    #Dby = Dby.to(device)
    Day = (1 - dt * sigmastary / 4) / (1 + dt * sigmastary / 4)
    #Day = Day.to(device)
    Cby = (dt / (1 + dt * sigmay / 2)) / dx
    #Cby = Cby.to(device)
    Cay = (1 - dt * sigmay / 2) / (1 + dt * sigmay / 2)
    #Cay = Cay.to(device)

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
        Jloader,
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


#@torch.jit.script
def sim(
    alpha0,
    ndelta,
    res,
    L,
    se,
    sb,
    vx,
    vy,
    x0,
    y0,
    Ef,
    Bf,
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
        Jloader,
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
    ) = sim_setup(alpha0, ndelta, res, se, sb, vx, vy, x0, y0, L)
    for J_x,J_y,J_z in Jloader:
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x * alpha,
            J_y * alpha,
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
    res,
    se,
    sb,
    vx,
    vy,
    alpha0,
    x0,
    y0,
    L,
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
        Jloader,
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
    ) = sim_setup(alpha0, ndelta, res, se, sb, vx, vy, x0, y0, L)
    nx = x.shape[0]
    Barr = torch.zeros((nx, nx, 3, t.shape[0]))
    Earr = torch.zeros((nx, nx, 3, t.shape[0]))
    for i,(J_x,J_y,J_z) in enumerate(Jloader):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x * alpha,
            J_y * alpha,
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
        Barr[..., 0, i] = b_x
        Barr[..., 1, i] = b_y
        Barr[..., 2, i] = b_zx + b_zy
        Earr[..., 0, i] = e_x
        Earr[..., 1, i] = e_y
        Earr[..., 2, i] = e_zx + e_zy
    return Barr.cpu(), Earr.cpu(), xx.cpu(), yy.cpu(), t.cpu()


def sim_bigbox(
    ndelta,
    res,
    se,
    sb,
    vx,
    vy,
    alpha0,
    x0,
    y0,
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
        Jloader,
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
    device = e_x.device
    for J_x,J_y,J_z in Jloader:
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
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
    Barr[..., 0] = b_x
    Barr[..., 1] = b_y
    Barr[..., 2] = b_zx + b_zy
    Earr[..., 0] = e_x
    Earr[..., 1] = e_y
    Earr[..., 2] = e_zx + e_zy

    return Barr.cpu(), Earr.cpu(), xx.cpu(), yy.cpu(), t.cpu()
