import torch

torch.set_default_dtype(torch.float32)


@torch.jit.script
def advance_b_x(b_x, e_z, Da, Db, mask):
    b_x = -1 * Db * torch.diff(e_z, dim=1, append=e_z[:, -1:]) + Da * b_x
    b_x *= mask
    return b_x


@torch.jit.script
def advance_b_y(b_y, e_z, Da, Db, mask):
    b_y = Db * torch.diff(e_z, dim=0, append=e_z[-1:, :]) + Da * b_y
    b_y *= mask
    return b_y


@torch.jit.script
def advance_b_zx(b_zx, e_y, Da, Db, mask):
    b_zx = -1 * Db * torch.diff(e_y, dim=0, append=e_y[-1:, :]) + Da * b_zx
    b_zx *= mask
    return b_zx


@torch.jit.script
def advance_b_zy(b_zy, e_x, Da, Db, mask):
    b_zy = Db * torch.diff(e_x, dim=1, append=e_x[:, -1:]) + Da * b_zy
    b_zy *= mask
    return b_zy


@torch.jit.script
def advance_e_x(e_x, b_z, J_x, dx, Ca, Cb, mask):
    e_x = (
        Cb * torch.roll(torch.diff(b_z, dim=1, append=mask[:, -1:]), 1, dims=1)
        + Ca * e_x
    )
    e_x *= mask
    e_x -= Cb * dx * J_x
    return e_x


@torch.jit.script
def advance_e_y(e_y, b_z, J_y, dy, Ca, Cb, mask):
    e_y = (
        -1 * Cb * torch.roll(torch.diff(b_z, dim=0, append=mask[-1:, :]), 1, dims=0)
        + Ca * e_y
    )
    e_y *= mask
    e_y -= Cb * dy * J_y
    return e_y


@torch.jit.script
def advance_e_zx(e_zx, b_y, J_z, dx, Ca, Cb, mask):
    e_zx = (
        Cb * torch.roll(torch.diff(b_y, dim=0, append=mask[-1:, :]), 1, dims=0)
        + Ca * e_zx
    )
    e_zx *= mask
    e_zx -= Cb * dx * J_z
    return e_zx


@torch.jit.script
def advance_e_zy(e_zy, b_x, J_z, dx, Ca, Cb, mask):
    e_zy = (
        -1 * Cb * torch.roll(torch.diff(b_x, dim=1, append=mask[:, -1:]), 1, dims=1)
        + Ca * e_zy
    )
    e_zy *= mask
    return e_zy


@torch.jit.script
def advance_b(e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy, Dax, Dbx, Day, Dby, maskb):
    b_x = advance_b_x(b_x, e_zx + e_zy, Day, Dby, maskb)
    b_y = advance_b_y(b_y, e_zx + e_zy, Dax, Dbx, maskb)
    b_zx = advance_b_zx(b_zx, e_y, Dax, Dbx, maskb)
    b_zy = advance_b_zy(b_zy, e_x, Day, Dby, maskb)
    return b_x, b_y, b_zx, b_zy


@torch.jit.script
def advance_e(
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
    Cay,
    Cby,
    maskex,
    maskey,
    maskez,
):
    e_x = advance_e_x(e_x, b_zx + b_zy, J_x, dx, Cay, Cby, maskex)
    e_y = advance_e_y(e_y, b_zx + b_zy, J_y, dx, Cax, Cbx, maskey)
    e_zx = advance_e_zx(e_zx, b_y, J_z, dx, Cax, Cbx, maskez)
    e_zy = advance_e_zy(e_zy, b_x, J_z, dx, Cay, Cby, maskez)
    return e_x, e_y, e_zx, e_zy


@torch.jit.script
def advance_flds(
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
):
    b_x, b_y, b_zx, b_zy = advance_b(
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy, Dax, Dbx, Day, Dby, maskb
    )
    e_x, e_y, e_zx, e_zy = advance_e(
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
        Cay,
        Cby,
        maskex,
        maskey,
        maskez,
    )
    b_x, b_y, b_zx, b_zy = advance_b(
        e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy, Dax, Dbx, Day, Dby, maskb
    )

    return e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy


@torch.jit.script
def masks(arr):
    maskb = torch.ones_like(arr)
    maskb[-1, :] = 0.0
    maskb[:, -1] = 0.0
    maskex = torch.ones_like(arr)
    maskex[:, 0] = 0.0
    maskex[:, -1] = 0.0
    maskey = torch.ones_like(arr)
    maskey[0, :] = 0.0
    maskey[-1, :] = 0.0
    maskez = torch.ones_like(arr)
    maskez[0, :] = 0.0
    maskez[-1, :] = 0.0
    maskez[:, 0] = 0.0
    maskez[:, -1] = 0.0
    return maskb, maskex, maskey, maskez
