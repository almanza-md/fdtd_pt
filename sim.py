import torch
from .grid import grid_setup, get_alpha, get_CD
from .fields import masks, advance_flds, field_arrs
from .current import jfunc

torch.set_default_dtype(torch.float32)


# @torch.jit.script
def sim_setup(
    ndelta,
    res,
    vx,
    vy,
    x0,
    y0,
    L,
):
    with torch.no_grad():
        x, xx, yy, delta, in_sim, dx = grid_setup(ndelta, res, L)
        device = ndelta.device
        J_x, J_y, t = jfunc(
            x.cpu(), vx, vy, L.cpu().to(torch.float32), x0=x0, y0=y0, delta=delta.cpu()
        )
        t = t.to(device)
        dt = t[1] - t[0]
        J_z = torch.zeros_like(J_x)

        J = torch.utils.data.TensorDataset(J_x, J_y, J_z)
        Jloader = torch.utils.data.DataLoader(
            J,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
        )

        maskb, maskex, maskey, maskez = masks(xx)

    return (
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
    )



def sim(
    alpha0,
    se,
    sb,
    xx,
    yy,
    ndelta,
    L,
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
):
    (e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy) = field_arrs(xx)

    (
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
    ) = get_CD(se, sb, xx, yy, ndelta, L, dt)
    alpha = get_alpha(alpha0, xx)
    device = xx.device
    for J_x, J_y, J_z in Jloader:
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            torch.squeeze(J_x.to(device)) * alpha,
            torch.squeeze(J_y.to(device)) * alpha,
            torch.squeeze(J_z.to(device)),
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
    alpha0,
    se,
    sb,
    t,
    xx,
    yy,
    ndelta,
    L,
    in_sim,
    Jloader,
    dx,
    dt,
    maskb,
    maskex,
    maskey,
    maskez,
):
    (e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy) = field_arrs(xx)

    (
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
    ) = get_CD(se, sb, xx, yy, ndelta, L, dt)
    alpha = get_alpha(alpha0, xx)
    device = xx.device
    Earr = torch.zeros((xx.shape[0], xx.shape[1], 3, t.shape[0]))
    Barr = torch.zeros((xx.shape[0], xx.shape[1], 3, t.shape[0]))
    for i, (J_x, J_y, J_z) in enumerate(Jloader):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            torch.squeeze(J_x.to(device)) * alpha,
            torch.squeeze(J_y.to(device)) * alpha,
            torch.squeeze(J_z.to(device)),
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
    se,
    sb,
    t,
    xx,
    yy,
    ndelta,
    L,
    Jloader,
    dx,
    dt,
    maskb,
    maskex,
    maskey,
    maskez,
):
    (e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy) = field_arrs(xx)

    (
        Cax,
        Cbx,
        Dax,
        Dbx,
        Cay,
        Cby,
        Day,
        Dby,
    ) = get_CD(se, sb, xx, yy, ndelta, L, dt)
    device = e_x.device
    Barr = torch.zeros((xx.shape[0], xx.shape[1], 3), device=device)
    Earr = torch.zeros((xx.shape[0], xx.shape[1], 3), device=device)

    for J_x, J_y, J_z in Jloader:
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            torch.squeeze(J_x.to(device)),
            torch.squeeze(J_y.to(device)),
            torch.squeeze(J_z.to(device)),
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

    return Barr, Earr, xx.cpu(), yy.cpu(), t.cpu()
