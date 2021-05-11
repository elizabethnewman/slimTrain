
import torch
import matplotlib.pyplot as plt
from networks.resnet import ResidualNetwork
from pinns.data import poisson2D
from pinns.poisson import PoissonPINNSlimTik
from pinns.training import train_sgd


# for reproducibility
torch.manual_seed(20)
torch.set_default_dtype(torch.float32)

# load data
d = 2       # domain dimension
n = 100    # number of interior points
nb = 10    # number of boundary points

a = 4.0     # constant
b = 2.0     # constant
data = poisson2D(d, n, nb, a, b)

# build network
feature_extractor = ResidualNetwork(in_features=d, width=10, depth=10, final_time=10,
                                    target_features=1, closing_layer=False)
W = torch.randn(1, 10 + 1)
pinn = PoissonPINNSlimTik(feature_extractor, W, bias=True, memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                          opt_method='none', reduction='sum', sumLambda=0.05)

# extract data
x, y, xb, yb, f, g = pinn.extract_data(data, requires_grad=True)

# "true" Laplacian
lap_u = pinn.net_f(x, y)

# check linearity
lap_z = pinn.lap_z(x, y)

if pinn.bias:
    lap_z = torch.cat((lap_z, torch.zeros(lap_z.shape[0], 1, dtype=lap_z.dtype)), dim=1)

lap_zWt = lap_z @ pinn.W.t()

print('|lap_u - lap_zWt| / |lap_u| = %0.4e' % (torch.norm(lap_u - lap_zWt) / torch.norm(lap_u)).item())