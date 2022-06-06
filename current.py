from operator import pos
import torch
import numpy as np
from math import sqrt, pow, atan2, sin, cos

torch.set_default_dtype(torch.float32)


def jfunc(x, vx, vy, L, x0, y0, delta, pml_dep=True, big_box=False):
    dx = x[1] - x[0]
    v = torch.sqrt(torch.square(vx) + torch.square(vy))
    radius = 0.055  # arbitrary
    theta = torch.atan2(vy, vx)
    wall_dist = L + delta
    if torch.isclose(vy, torch.zeros(1)):
        dist = (wall_dist - x0) / torch.cos(theta)
        pml_dist = (L - x0) / torch.cos(theta)
    elif torch.isclose(vx, torch.zeros(1)):
        dist = (wall_dist - y0) / torch.abs(torch.sin(theta))
        pml_dist = (L - y0) / torch.abs(torch.sin(theta))
    else:
        dist = torch.min(
            torch.stack(
                (
                    (wall_dist - x0) / torch.cos(theta),
                    (wall_dist - y0) / torch.abs(torch.sin(theta)),
                )
            )
        )
        pml_dist = torch.min(
            torch.stack(
                ((L - x0) / torch.cos(theta), (L - y0) / torch.abs(torch.sin(theta)))
            )
        )
    jtmax = dist / v
    nodep_tmax = pml_dist / v
    tmax = float(jtmax + 2 * sqrt(2.0) * delta)
    dt = float(0.98 * dx / sqrt(2))
    print(tmax)
    t = torch.arange(start=0, end=tmax, step=dt)
    tt, xx, yy = torch.meshgrid(t, x, x, indexing="ij")
    c_shape = torch.exp(
        -1
        * (torch.square(xx - x0 - vx * tt) + torch.square(yy - y0 - vy * tt))
        / pow(radius, 2)
    ) / (pow(radius * torch.pi, 2))
    # c_shape[torch.isclose(c_shape, torch.zeros_like(c_shape),atol=1e-10)] *= 0
    c_weight = torch.ones_like(tt)
    if big_box:
        c_weight[tt > tmax] = 0.0
    elif pml_dep:
        c_weight[tt > jtmax] = 0.0
    else:
        c_weight[tt > nodep_tmax] = 0.0
    Jx = torch.zeros_like(tt)
    Jy = torch.zeros_like(tt)
    Jx += c_weight * c_shape * vx
    Jy += c_weight * c_shape * vy
    return Jx, Jy, t


def particle_shape(pos, x, y, q=1):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xidx = torch.argmin(torch.abs(pos[:, 0] - x), dim=0)
    yidx = torch.argmin(torch.abs(pos[:, 1] - y), dim=0)
    px = torch.diag((pos[:, 0] - x[xidx]) / dx)
    #px += torch.sign(px) * 0.5
    py = torch.diag((pos[:, 1] - y[yidx]) / dy)
    #py += torch.sign(py) * 0.5
    Sx = torch.unsqueeze((q / 1) * (3 / 4 - torch.square(px)), dim=1)
    pad = torch.zeros_like(Sx)
    Sx_m = torch.unsqueeze((q / (2 * 1)) * torch.square((1 / 2 - px)), dim=1)
    Sx_p = torch.unsqueeze((q / (2 * 1)) * torch.square((1 / 2 + px)), dim=1)
    Sy = torch.unsqueeze((q / 1) * (3 / 4 - torch.square(py)), dim=1)
    Sy_m = torch.unsqueeze((q / (2 * 1)) * torch.square((1 / 2 - py)), dim=1)
    Sy_p = torch.unsqueeze((q / (2 * 1)) * torch.square((1 / 2 + py)), dim=1)
    Sx_prof = torch.stack((pad, Sx_m, Sx, Sx_p, pad), dim=0)
    Sy_prof = torch.stack((pad, Sy_m, Sy, Sy_p, pad), dim=0)
    return (
        xidx,
        yidx,
        Sx_prof,
        Sy_prof,
    )


def Wfunc(Sx_old, Sy_old, DSx, DSy):
    W = torch.zeros(Sx_old.shape[0], Sx_old.shape[0], Sx_old.shape[1], 1, 3)
    W[..., 0] = torch.einsum("i...,j...->ij...", DSx, (Sy_old + 0.5 * DSy))
    W[..., 1] = torch.einsum("i...,j...->ij...", (Sx_old + 0.5 * DSx), DSy)
    W[..., 2] = (
        torch.einsum("i...,j...->ij...", Sx_old, Sy_old)
        + torch.einsum("i...,j...->ij...", DSx, DSy) / 3
        + 0.5 * torch.einsum("i...,j...->ij...", Sx_old, DSy)
        + 0.5 * torch.einsum("i...,j...->ij...", DSx, Sy_old)
    )
    return W


# from fdtd_pt.current import current_dep
def current_dep(pos, vel, xx, yy, q=1, init=False,old_pos=None,old_vel=None):
    # if J0 is None:
    J = torch.zeros((xx.shape[0], xx.shape[1], 3), device=xx.device)
    x = xx[:, 0:1]
    n = x.shape[0]
    dx = x[1] - x[0]
    y = torch.unsqueeze(torch.squeeze(yy[0:1, :]), dim=1)
    m = y.shape[0]
    dy = y[1] - y[0]
    dt = float(0.98 * dx / sqrt(2))
    
    xidx, yidx, Sx_new, Sy_new = particle_shape(pos, x, y, q)
    if old_pos is not None:
        xidx_old, yidx_old, Sx_old, Sy_old = particle_shape(
            old_pos, x, y, q
        )
    else:
        xidx_old, yidx_old, Sx_old, Sy_old = particle_shape(
            pos - dt * vel[:, 0:2], x, y, q
        )
    if old_vel is not None:
        vel = 0.5*(vel+old_vel)
    xshifts = xidx - xidx_old
    yshifts = yidx - yidx_old
    # print(xshifts)

    if init:
        Sxy = torch.einsum("i...,j...->ij...", Sx_new, Sy_new)
        J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, :] += torch.squeeze(Sxy*vel)
        # J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, 1] += torch.squeeze(W2)
        # J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, 2] += torch.squeeze(W3)
        # J[xidx, yidx, :] += Sx_new[2] * Sy_new[2] * vel
        # J[xidx - 1, yidx, :] += Sx_new[1] * Sy_new[2] * vel
        # J[xidx - 1, yidx - 1, :] += Sx_new[1] * Sy_new[1] * vel
        # J[xidx, yidx - 1, :] += Sx_new[2] * Sy_new[1] * vel
        # J[(xidx + 1) % n, yidx, :] += Sx_new[3] * Sy_new[2] * vel
        # J[xidx, (yidx + 1) % m, :] += Sx_new[2] * Sy_new[3] * vel
        # J[(xidx + 1) % n, (yidx + 1) % m, :] += Sx_new[3] * Sy_new[3] * vel
        # J[xidx - 1, (yidx + 1) % m, :] += Sx_new[1] * Sy_new[3] * vel
        # J[(xidx + 1) % n, yidx - 1, :] += Sx_new[3] * Sy_new[1] * vel
        return J
    else:
        Sx_new = torch.squeeze(
            torch.stack(
                [
                    torch.roll(s, int(xs), dims=0)
                    for (s, xs) in zip(torch.split(Sx_new, 1, dim=1), xshifts)
                ],
                dim=1,
            ),
            dim=-1,
        )
        DSx = Sx_new - Sx_old
        Sy_new = torch.squeeze(
            torch.stack(
                [
                    torch.roll(s, int(ys), dims=0)
                    for (s, ys) in zip(torch.split(Sy_new, 1, dim=1), yshifts)
                ],
                dim=1,
            ),
            dim=-1,
        )
        DSy = Sy_new - Sy_old

        W = Wfunc(Sx_old, Sy_old, DSx, DSy)

        W1 = -q * (dx/dt) * torch.cumsum(W[..., 0], dim=0)
        W2 = -q * (dy/dt) * torch.cumsum(W[..., 1], dim=1)
        W3 = -q * vel[:, 2] * W[..., 2]

        #W1 = q*(dx/dt)*(W[...,0] +  torch.sum(W[...,0],dim=0, keepdims=True) - torch.cumsum(W[...,0],dim=0))
        #W2 = q*(dy/dt)*(W[...,1] +  torch.sum(W[...,1],dim=1, keepdims=True) - torch.cumsum(W[...,1],dim=1))
        #W3 = q*vel[:,2]*W[...,2]
        # print(DSx)
        # print(Sx_old.shape)
        try:
            J[
                xidx_old - 2 : xidx_old + 3, yidx_old - 2 : yidx_old + 3, 0
            ] += torch.squeeze(W1)
            J[
                xidx_old - 2 : xidx_old + 3, yidx_old - 2 : yidx_old + 3, 1
            ] += torch.squeeze(W2)
            J[
                xidx_old - 2 : xidx_old + 3, yidx_old - 2 : yidx_old + 3, 2
            ] += torch.squeeze(W3)
        except:
            pass
        # J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, 0] += torch.squeeze(W1)
        # J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, 1] += torch.squeeze(W2)
        # J[xidx - 2 : xidx + 3, yidx - 2 : yidx + 3, 2] += torch.squeeze(W3)
        # J[xidx,yidx,:]                 += Sx_new[2]*Sy_new[2]*vel
        # J[xidx-1,yidx,:]               += Sx_new[1]*Sy_new[2]*vel
        # J[xidx-1,yidx-1,:]             += Sx_new[1]*Sy_new[1]*vel
        # J[xidx,yidx-1,:]               += Sx_new[2]*Sy_new[1]*vel
        # J[(xidx+1) % n,yidx,:]         += Sx_new[3]*Sy_new[2]*vel
        # J[xidx,(yidx+1) % m,:]         += Sx_new[2]*Sy_new[3]*vel
        # J[(xidx+1) % n,(yidx+1) % m,:] += Sx_new[3]*Sy_new[3]*vel
        # J[xidx-1,(yidx+1) % m,:]       += Sx_new[1]*Sy_new[3]*vel
        # J[(xidx+1) % n,yidx-1,:]       += Sx_new[3]*Sy_new[1]*vel
    return J


# from fdtd_pt.current import current_dep
def current_dep_old(pos, vel, xx, yy):
    J = torch.zeros((xx.shape[0], xx.shape[1], 3), device=xx.device)
    x = xx[:, 0:1]
    n = x.shape[0]
    dx = x[1] - x[0]
    y = torch.squeeze(yy[0:1, :])
    m = y.shape[0]
    dy = y[1] - y[0]
    xidx = torch.argmin(torch.abs(pos[:, 0] - x), dim=0)
    yidx = torch.argmin(torch.abs(pos[:, 1] - y), dim=0)
    px = (pos[:, 0] - x[xidx]) / dx
    py = (pos[:, 1] - y[yidx]) / dy
    Sx = (1 / dx) * (3 / 4 - torch.square(px))
    Sx_m = (1 / (2 * dx)) * torch.square((1 / 2 - px))
    Sx_p = (1 / (2 * dx)) * torch.square((1 / 2 + px))
    Sy = (1 / dy) * (3 / 4 - torch.square(py))
    Sy_m = (1 / (2 * dy)) * torch.square((1 / 2 - py))
    Sy_p = (1 / (2 * dy)) * torch.square((1 / 2 + py))
    J[xidx, yidx, :] += Sx * Sy * vel
    J[xidx - 1, yidx, :] += Sx_m * Sy * vel
    J[xidx - 1, yidx - 1, :] += Sx_m * Sy_m * vel
    J[xidx, yidx - 1, :] += Sx * Sy_m * vel
    J[(xidx + 1) % n, yidx, :] += Sx_p * Sy * vel
    J[xidx, (yidx + 1) % m, :] += Sx * Sy_p * vel
    J[(xidx + 1) % n, (yidx + 1) % m, :] += Sx_p * Sy_p * vel
    J[xidx - 1, (yidx + 1) % m, :] += Sx_m * Sy_p * vel
    J[(xidx + 1) % n, yidx - 1, :] += Sx_p * Sy_m * vel

    return J


def jfunc_dep(x, vx, vy, L, x0, y0, delta, pml_dep=True, big_box=False,smooth=False,filter_n=1,filter_mode='bilinear'):
    assert filter_mode.lower() in ['bilinear', 'gaussian']
    dx = x[1] - x[0]
    # v = torch.sqrt(torch.square(vx) + torch.square(vy))
    v = sqrt((vx) ** 2 + (vy) ** 2)
    theta = atan2(vy, vx)
    # L-=dx
    wall_dist = L + delta
    if np.isclose(vy, 0):
        dist = (wall_dist - x0) / cos(theta)
        pml_dist = (L - x0) / cos(theta)
    elif np.isclose(vx, 0):
        dist = (wall_dist - y0) / abs(sin(theta))
        pml_dist = (L - y0) / abs(sin(theta))
    else:
        dist = min(((wall_dist - x0) / cos(theta), (wall_dist - y0) / abs(sin(theta))))
        pml_dist = min(((L - x0) / cos(theta), (L - y0) / abs(sin(theta))))
    dt = float(0.98 * dx / sqrt(2))
    jtmax = dist / v
    nodep_tmax = pml_dist / v
    tmax = float(jtmax + 2 * sqrt(2.0) * delta)
    t = torch.arange(start=0, end=tmax, step=dt)
    tt, xx, yy = torch.meshgrid(t, x, x, indexing="ij")
    c_weight = torch.ones_like(tt)

    if big_box:
        c_weight[tt > tmax] = 0.0
        t0 = tmax
    elif pml_dep:
        c_weight[tt > jtmax] = 0.0
        t0 = jtmax
    else:
        c_weight[tt > nodep_tmax] = 0.0
        t0 = nodep_tmax
    vel = torch.tensor([[vx, vy, 0.0]])
    #vel_list = [vel * 0.5*(1+torch.tanh(1*(ti-2))) for ti in t]
    vel_list = [vel * sin(ti*torch.pi/2)**2 if ti<1 else vel for ti in t]
    poslist = [torch.tensor([[x0 + v[0, 0] * tp, y0 + v[0, 1] * tp]]) for (tp,v) in zip(t,vel_list)]
    poslist = [torch.tensor([[x0,y0]])]
    for i in range(1,len(t)):
        dx = 0.5*(vel_list[i]+vel_list[i-1])*dt
        poslist.append(poslist[i-1] + dx[:,0:2])
    Jlist = [current_dep(poslist[0], 0*vel_list[0], xx[0, ...], yy[0, ...], init=True),
    current_dep(poslist[0], vel_list[0], xx[0, ...], yy[0, ...], old_pos=poslist[0],old_vel=vel_list[0])]
    ptemp = poslist[0]
    for i, tp in zip(range(2, len(t)), t[:-1]):
        if tp < t0:
            Jlist.append(current_dep(poslist[i-1], vel_list[i-1], xx[0, ...], yy[0, ...],old_pos=poslist[i-2],old_vel=vel_list[i-2]))
        #elif tp == t0:
        #    Jlist.append(current_dep(poslist[i-1], vel_list[i-1], xx[0, ...], yy[0, ...],old_pos=poslist[i-2],old_vel=vel_list[i-2]))
        #    ptemp = poslist[i]
        else:
            Jlist.append(current_dep(poslist[i-1], 0 * vel_list[i-1], xx[0, ...], yy[0, ...],old_pos=poslist[i-2],old_vel=0*vel_list[i-2]))

    Jtensor = torch.stack(Jlist, dim=0)
    if filter_mode=='gaussian':
        filter_range = torch.linspace(-filter_n,filter_n,2*filter_n+1)
        fxx,fyy = torch.meshgrid(filter_range,filter_range,indexing='ij')
        gfilter = torch.exp(-1/2 * (torch.square(fxx)+torch.square(fyy)))
        filter = torch.reshape(gfilter,(1,1,2*filter_n+1,2*filter_n+1))
    else:
        filter = torch.tensor([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
        filter = torch.reshape(filter,(1,1,3,3))
    # Jtensor /= torch.max(Jtensor)
    # Jtensor *= 4 * np.pi
    Jx, Jy, Jz = [
        torch.clone(torch.squeeze(jj)) for jj in torch.split(Jtensor, 1, dim=-1)
    ]
    if smooth:
        Jx = torch.squeeze(torch.nn.functional.conv2d(torch.unsqueeze(Jx,dim=1),filter,padding='same'))
        Jy = torch.squeeze(torch.nn.functional.conv2d(torch.unsqueeze(Jy,dim=1),filter,padding='same'))
        Jz = torch.squeeze(torch.nn.functional.conv2d(torch.unsqueeze(Jz,dim=1),filter,padding='same'))
    #Jx *= c_weight
    #Jy *= c_weight
    #Jz *= c_weight
    return Jx, Jy, Jz, t
