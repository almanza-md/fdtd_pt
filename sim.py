import torch
from .grid import grid_setup, get_alpha, get_CD
from .fields import masks, advance_flds, field_arrs
from .current import jfunc_dep
from math import sqrt

torch.set_default_dtype(torch.float32)


# @torch.jit.script
def sim_setup_old(
    ndelta,
    res,
    vx,
    vy,
    x0,
    y0,
    L,
    L0=torch.tensor(2),
    use_delta=True,
    smooth=False,
    filter_n=1,
    filter_mode="bilinear",
):
    with torch.no_grad():
        x, xx, yy, delta, in_sim, dx = grid_setup(ndelta, res, L)
        device = ndelta.device
        J_x, J_y, J_z, t = jfunc_dep(
            x.cpu(),
            vx,
            vy,
            L0.cpu().to(torch.float32),
            x0=x0,
            y0=y0,
            delta=delta.cpu(),
            pml_dep=use_delta,
            big_box=(L > 2),
            smooth=smooth,
            filter_n=filter_n,
            filter_mode=filter_mode,
        )
        t = t.to(device)
        dt = t[1] - t[0]
        # J_z = torch.zeros_like(J_x)

        J = torch.utils.data.TensorDataset(J_x, J_y, J_z)
        Jloader = torch.utils.data.DataLoader(
            J,
            num_workers=4,
            pin_memory=True,
            # prefetch_factor=8,
            persistent_workers=True,
            batch_size=None,
        )

        maskb, maskex, maskey, maskez = masks(xx)
        (e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy) = field_arrs(xx)
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
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
    )


def sim_setup(
    ndelta,
    res,
    vx,
    vy,
    x0,
    y0,
    Lx,
    Ly,
    L0=None,
    t_max=None,
    use_delta=True,
    smooth=False,
    filter_n=1,
    filter_mode="bilinear",
    sparse_j=False,
):
    with torch.no_grad():
        x, y, xx, yy, delta, in_sim, dx = grid_setup(ndelta, res, Lx, Ly)
        device = ndelta.device
        big_box = False
        if L0 is not None:
            Lx = L0[0]
            Ly = L0[1]
            big_box = True
        J_x1, J_y1, J_z1, t1 = jfunc_dep(
            x.cpu(),
            y.cpu(),
            vx,
            vy,
            Lx,
            Ly,
            x0=x0,
            y0=y0,
            delta=delta.cpu(),
            pml_dep=use_delta,
            big_box=big_box,
            t_max=t_max,
            smooth=smooth,
            filter_n=filter_n,
            filter_mode=filter_mode,
        )
        J_x2, J_y2, J_z2, t2 = jfunc_dep(
            x.cpu(),
            y.cpu(),
            -vx,
            -vy,
            Lx,
            Ly,
            x0=x0,
            y0=y0,
            delta=delta.cpu(),
            pml_dep=use_delta,
            big_box=big_box,
            smooth=smooth,
            filter_n=filter_n,
            filter_mode=filter_mode,
            q=-1,
        )
        l1 = len(t1)
        l2 = len(t2)
        if l2 > l1:
            J_x = J_x2
            J_x[0:l1, ...] += J_x1
            J_y = J_y2
            J_y[0:l1, ...] += J_y1
            J_z = J_z2
            J_z[0:l1, ...] += J_z1
            t = t2.to(device)
        else:
            J_x = J_x1
            J_x[0:l2, ...] += J_x2
            J_y = J_y1
            J_y[0:l2, ...] += J_y2
            J_z = J_z1
            J_z[0:l2, ...] += J_z2
            t = t1.to(device)
        dt = t[1] - t[0]
        # J_z = torch.zeros_like(J_x)

        J = torch.utils.data.TensorDataset(J_x, J_y, J_z)
        Jloader = torch.utils.data.DataLoader(
            J,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True,
            batch_size=None,
        )


        maskb, maskex, maskey, maskez = masks(xx)
        (e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy) = field_arrs(xx)
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
        e_x,
        e_y,
        e_zx,
        e_zy,
        b_x,
        b_y,
        b_zx,
        b_zy,
    )


def sim(
    alpha0,
    se,
    sb,
    t,
    xx,
    yy,
    ndelta,
    Lx,
    Ly,
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
    Bf
):

    (Dbx, Dax, Cbx, Cax, Dby, Day, Cby, Cay) = get_CD(
        se, sb, xx, yy, ndelta, Lx, Ly, dt
    )
    alpha = get_alpha(alpha0, xx)
    if type(alpha) == tuple:
        alphax = alpha[0]
        alphay = alpha[1]
        # alphaz = alpha[2]
    else:
        alphax = alpha
        alphay = alpha
        # alphaz = alpha
    device = xx.device
    n = int(Ef.shape[-1])
    Barr = torch.zeros((xx.shape[0], xx.shape[1], 3, n), device=device)
    Earr = torch.zeros((xx.shape[0], xx.shape[1], 3, n), device=device)
    nt = t.shape[0]
    d = float(dx*ndelta)
    tminidx = torch.argmin(torch.abs(t-(t[-1]-(2*sqrt(2)-1)*d)))
    tpts = torch.linspace(tminidx,nt-1,n,dtype=torch.int64)
    if n==1:
        tpts = torch.tensor([nt-1])
    for i,(J_x, J_y, J_z) in enumerate(Jloader):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x.to(device) * alphax,
            J_y.to(device) * alphay,
            J_z.to(device),  # * alphaz,
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
        if i in tpts:
            ni = int(torch.argwhere(tpts==i)[0])
            Earr[...,ni] = torch.stack((e_x, e_y, e_zx + e_zy), dim=-1)
            Barr[...,ni] = torch.stack((b_x, b_y, b_zx + b_zy), dim=-1)
    # Eerr = torch.nn.functional.l1_loss(Earr,Ef)
    # Berr = torch.nn.functional.l1_loss(Barr,Bf)
    # Utot = Eerr+Berr
    u = torch.sum(torch.square(Earr - Ef) + torch.square(Barr - Bf), dim=2)
    u *= torch.unsqueeze(in_sim,dim=-1)
    #Utot = torch.sum(u)
    Utot = torch.trapezoid(torch.trapezoid(u,x=yy[0,:],dim=1),x=xx[:,0],dim=0)
    Utot = torch.mean(Utot)
    return Utot


def sim_EB(
    alpha0,
    se,
    sb,
    t,
    xx,
    yy,
    ndelta,
    Lx,
    Ly,
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
):

    (Dbx, Dax, Cbx, Cax, Dby, Day, Cby, Cay) = get_CD(
        se, sb, xx, yy, ndelta, Lx, Ly, dt
    )
    alpha = get_alpha(alpha0, xx)
    if type(alpha) == tuple:
        alphax = alpha[0]
        alphay = alpha[1]
        # alphaz = alpha[2]
    else:
        alphax = alpha
        alphay = alpha
        # alphaz = alpha
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
            J_x.to(device) * alphax,
            J_y.to(device) * alphay,
            J_z.to(device),
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
    Lx,
    Ly,
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
    n=1
):

    (Dbx, Dax, Cbx, Cax, Dby, Day, Cby, Cay) = get_CD(
        se, sb, xx, yy, ndelta, Lx, Ly, dt
    )
    device = e_x.device
    Barr = torch.zeros((xx.shape[0], xx.shape[1], 3, n), device=device)
    Earr = torch.zeros((xx.shape[0], xx.shape[1], 3, n), device=device)
    nt = t.shape[0]
    d = float(dx*ndelta)
    tminidx = torch.argmin(torch.abs(t-(t[-1]-(2*sqrt(2)-1)*d)))
    tpts = torch.linspace(tminidx,nt-1,n,dtype=torch.int64)
    if n==1:
        tpts = torch.tensor([nt-1])
    for i,(J_x, J_y, J_z) in enumerate(Jloader):
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy = advance_flds(
            e_x,
            e_y,
            e_zx,
            e_zy,
            b_x,
            b_y,
            b_zx,
            b_zy,
            J_x.to(device),
            J_y.to(device),
            J_z.to(device),
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
        if i in tpts:
            ni = int(torch.argwhere(tpts==i)[0])
            Barr[..., 0, ni] = b_x
            Barr[..., 1, ni] = b_y
            Barr[..., 2, ni] = b_zx + b_zy
            Earr[..., 0, ni] = e_x
            Earr[..., 1, ni] = e_y
            Earr[..., 2, ni] = e_zx + e_zy

    return Barr, Earr, xx.cpu(), yy.cpu(), t.cpu(), tpts
