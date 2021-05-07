import torch
from math import pi


def poisson2D(d=2, n=1000, nb=100, a=4.0, b=2.0):

    # get points in domain
    X = (torch.rand((n, d)) - 0.5) * 2
    X = X.detach()

    # get points on boundary
    xx = torch.linspace(-1, 1, nb).reshape(1, nb)
    ee = torch.ones_like(xx)
    Xb = []
    Xb.append(torch.cat((xx, ee), 0))
    Xb.append(torch.cat((xx, -ee), 0))
    Xb.append(torch.cat((ee, xx), 0))
    Xb.append(torch.cat((-ee, xx), 0))
    Xb = torch.cat(Xb, 1)
    Xb = Xb.t().detach()

    u_true = torch.sin(a * pi * X[:, 0]) * torch.sin(b * pi * X[:, 1])
    f = -(a**2 + b**2) * (pi**2) * u_true
    ub_true = torch.sin(a * pi * Xb[:, 0]) * torch.sin(b * pi * Xb[:, 1])

    data = {'interior': X, 'boundary': Xb, 'forcing': f,
            'interior_true': u_true, 'boundary_true': ub_true,
            'constants_true': (a, b)}

    return data
