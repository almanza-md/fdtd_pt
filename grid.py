import torch

torch.set_default_dtype(torch.float32)


@torch.jit.script
def grid_setup(ndelta, res, L=torch.tensor(2)):
    device = ndelta.device
    nx = 2 * L * res + 1  # torch.floor(2*L*res).to(torch.int64)
    L = L.to(device)
    x = torch.linspace(-L, L, nx.item())

    dx = x[1] - x[0]
    pmlx = torch.stack([L + n * dx for n in torch.arange(1, ndelta.item() + 1)])

    x = torch.cat((-1 * torch.flipud(pmlx), x, pmlx))
    x = x.to(device=device)
    dx = x[1] - x[0]
    nx = x.shape[0]
    ny = nx
    y = x
    delta = ndelta * dx
    e_x = torch.zeros((nx, ny), device=device)
    e_y = torch.zeros((nx, ny), device=device)
    e_zx = torch.zeros((nx, ny), device=device)
    e_zy = torch.zeros((nx, ny), device=device)
    b_x = torch.zeros((nx, ny), device=device)
    b_y = torch.zeros((nx, ny), device=device)
    b_zx = torch.zeros((nx, ny), device=device)
    b_zy = torch.zeros((nx, ny), device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return x, xx, yy, delta, e_x, e_y, e_zx, e_zy, b_x, b_y, b_zx, b_zy
