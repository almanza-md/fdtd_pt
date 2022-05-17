import torch
from math import sqrt,pow

torch.set_default_dtype(torch.float32)


def jfunc(x, vx, vy, L, x0, y0, delta, pml_dep=True):
    dx = x[1] - x[0]
    v = torch.sqrt(torch.square(vx) + torch.square(vy))
    radius = 0.055 # arbitrary
    theta = torch.atan2(vy, vx)
    if pml_dep:
        L += delta
    if torch.isclose(vy, torch.zeros(1)):
        dist = (L - x0) / torch.cos(theta)
    elif torch.isclose(vx, torch.zeros(1)):
        dist = (L - y0) / torch.sin(theta)
    else:
        dist = torch.min(
            torch.stack(((L - x0) / torch.cos(theta), (L - y0) / torch.sin(theta)))
        )
    jtmax = dist / v
    tmax = float(jtmax + 2 * sqrt(2.0) * delta)
    dt = float(0.98 * dx / sqrt(2))
    t = torch.arange(start=0, end=tmax, step=dt)
    tt, xx, yy = torch.meshgrid(t, x, x, indexing="ij")
    c_shape = torch.exp(
        -1
        * (torch.square(xx - x0 - vx * tt) + torch.square(yy - y0 - vy * tt))
        / pow(radius,2)
    ) / (pow(radius * torch.pi,2))
    c_shape[torch.isclose(c_shape, torch.zeros_like(c_shape))] *= 0
    c_weight = torch.ones_like(tt)
    c_weight[tt > tmax] = 0.0
    Jx = torch.zeros_like(tt)
    Jy = torch.zeros_like(tt)
    Jx += c_weight * c_shape * vx
    Jy += c_weight * c_shape * vy
    return Jx, Jy, t
