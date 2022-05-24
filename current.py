import torch
import numpy as np
from math import sqrt,pow, atan2, sin, cos

torch.set_default_dtype(torch.float32)


def jfunc(x, vx, vy, L, x0, y0, delta, pml_dep=True, big_box=False):
    dx = x[1] - x[0]
    v = torch.sqrt(torch.square(vx) + torch.square(vy))
    radius = 0.055 # arbitrary
    theta = torch.atan2(vy, vx)
    wall_dist = L+delta
    if torch.isclose(vy, torch.zeros(1)):
        dist = (wall_dist - x0) / torch.cos(theta)
        pml_dist = (L - x0) / torch.cos(theta)
    elif torch.isclose(vx, torch.zeros(1)):
        dist = (wall_dist - y0) / torch.sin(theta)
        pml_dist = (L - y0) / torch.sin(theta)
    else:
        dist = torch.min(
            torch.stack(((wall_dist - x0) / torch.cos(theta), (wall_dist - y0) / torch.sin(theta)))
        )
        pml_dist = torch.min(
            torch.stack(((L - x0) / torch.cos(theta), (L - y0) / torch.sin(theta)))
        )
    jtmax = dist / v
    nodep_tmax = pml_dist/v
    tmax = float(jtmax + 2 * sqrt(2.0) * delta)
    dt = float(0.98 * dx / sqrt(2))
    t = torch.arange(start=0, end=tmax, step=dt)
    tt, xx, yy = torch.meshgrid(t, x, x, indexing="ij")
    c_shape = torch.exp(
        -1
        * (torch.square(xx - x0 - vx * tt) + torch.square(yy - y0 - vy * tt))
        / pow(radius,2)
    ) / (pow(radius * torch.pi,2))
    #c_shape[torch.isclose(c_shape, torch.zeros_like(c_shape),atol=1e-10)] *= 0
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

#from fdtd_pt.current import current_dep
def current_dep(pos,vel,xx,yy):
    J = torch.zeros((xx.shape[0],xx.shape[1],3),device=xx.device)
    x = xx[:,0:1]
    n = x.shape[0]
    dx = x[1] - x[0]
    y = torch.squeeze(yy[0:1,:])
    m=y.shape[0]
    dy = y[1]-y[0]
    xidx = torch.argmin(torch.abs(pos[:,0]-x),dim=0)
    yidx = torch.argmin(torch.abs(pos[:,1]-y),dim=0)
    px = (pos[:,0] - x[xidx])/dx
    py = (pos[:,1] - y[yidx])/dy
    Sx = (1/dx)*(3/4 - torch.square(px))
    Sx_m = (1/(2*dx)) * torch.square((1/2 - px))
    Sx_p = (1/(2*dx)) * torch.square((1/2 + px))
    Sy = (1/dy)*(3/4 - torch.square(py))
    Sy_m = (1/(2*dy)) * torch.square((1/2 - py))
    Sy_p = (1/(2*dy)) * torch.square((1/2 + py))
    J[xidx,yidx,:]     += Sx*Sy*vel
    J[xidx-1,yidx,:]   += Sx_m*Sy*vel
    J[xidx-1,yidx-1,:] += Sx_m*Sy_m*vel
    J[xidx,yidx-1,:]   += Sx*Sy_m*vel
    J[(xidx+1) % n,yidx,:]   += Sx_p*Sy*vel
    J[xidx,(yidx+1) % m,:]   += Sx*Sy_p*vel
    J[(xidx+1) % n,(yidx+1) % m,:] += Sx_p*Sy_p*vel
    J[xidx-1,(yidx+1) % m,:] += Sx_m*Sy_p*vel
    J[(xidx+1) % n,yidx-1,:] += Sx_p*Sy_m*vel

    return J

def jfunc_dep(x, vx, vy, L, x0, y0, delta, pml_dep=True, big_box=False):
    dx = x[1] - x[0]
    #v = torch.sqrt(torch.square(vx) + torch.square(vy))
    v = sqrt(vx**2+vy**2)
    theta = atan2(vy, vx)
    #L-=dx
    wall_dist = L+delta
    if np.isclose(vy, 0):
        dist = (wall_dist - x0) / cos(theta)
        pml_dist = (L - x0) / cos(theta)
    elif np.isclose(vx, 0):
        dist = (wall_dist - y0) / sin(theta)
        pml_dist = (L - y0) / sin(theta)
    else:
        dist = min(
            ((wall_dist - x0) / cos(theta), (wall_dist - y0) / sin(theta))
        )
        pml_dist = min(
            ((L - x0) / cos(theta), (L - y0) / sin(theta))
        )
    dt = float(0.98 * dx / sqrt(2))
    jtmax = dist / v
    nodep_tmax = pml_dist/v
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
    vel = torch.tensor([[vx,vy,0.]])
    poslist = [torch.tensor([[x0+vel[0,0]*tp,y0+vel[0,1]*tp]]) for tp in t]
    Jlist = [current_dep(pos,vel,xx[0,...],yy[0,...]) if tp<t0 else current_dep(pos,0*vel,xx[0,...],yy[0,...]) for (pos,tp) in zip(poslist,t)]
    Jtensor = torch.stack(Jlist,dim=0)
    Jtensor /= torch.max(Jtensor)
    Jtensor *= 4*np.pi
    Jx,Jy,_= [torch.clone(torch.squeeze(jj)) for jj in torch.split(Jtensor,1,dim=-1)]
    
    Jx *= c_weight
    Jy *= c_weight
    return Jx, Jy, t



    
