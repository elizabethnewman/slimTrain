import torch
import matplotlib.pyplot as plt
from networks.resnet import ResidualNetwork
from pinns.data import poisson2D
from pinns.poisson import PoissonPINN
from pinns.training import train_lbfgs

# for saving
import shutil
import datetime
import sys

# for reproducibility
torch.manual_seed(20)
torch.set_default_dtype(torch.float64)

# load data
d = 2       # domain dimension
n = 1000    # number of interior points
nb = 100    # number of boundary points

a = 4.0     # constant
b = 2.0     # constant
data = poisson2D(d, n, nb, a, b)

# build network
feature_extractor = ResidualNetwork(in_features=d, width=10, depth=10, final_time=10,
                                    target_features=1, closing_layer=True)
pinn = PoissonPINN(feature_extractor)

# optimization
opt = torch.optim.LBFGS(feature_extractor.parameters(), line_search_fn='strong_wolfe', max_iter=100)
results = train_lbfgs(pinn, opt, data, 50)

# save!
# filename = 'tmp'
# torch.save((pinn.net_u.state_dict(), results), 'results/' + filename + '.pt')
# shutil.copy(sys.argv[0], 'results/' + filename + '.py')

# plotting
u_true = data['interior_true']
x, y, xb, yb, f, g = pinn.extract_data(data, requires_grad=True)

u = pinn.net_u(torch.cat((x, y), dim=1))
f_pred = pinn.net_f(x, y)
ub = pinn.net_u(torch.cat((xb, yb), dim=1))

print('u: ', torch.norm(u.view(-1) - u_true.view(-1)) / torch.norm(u_true))
print('lapu: ', torch.norm(f.view(-1) - f_pred.view(-1)) / torch.norm(f))
print('boundary: ', torch.norm(ub.view(-1) - g.view(-1)))

# plot PDE
plt.figure(1)
plt.subplot(2, 2, 1)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=f.numpy())
plt.colorbar()
plt.title("lapu true")

plt.subplot(2, 2, 2)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=f_pred.detach().numpy())
plt.colorbar()
plt.title("lapu approx")

plt.subplot(2, 2, 4)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=torch.abs(f_pred.detach().view(-1) - f.view(-1)).numpy())
plt.colorbar()
plt.title("diff")

plt.show()

# plot solution
plt.figure(2)
plt.subplot(2, 2, 1)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u_true.numpy())
plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=g.detach().numpy())
plt.colorbar()
plt.title("utrue")

plt.subplot(2, 2, 2)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u.detach().view(-1).numpy())
plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=ub.detach().view(-1).numpy())
plt.colorbar()
plt.title("approx")

plt.subplot(2, 2, 3)
plt.semilogy(results['val'][:, results['str'].index('loss')].numpy())
plt.semilogy(results['val'][:, results['str'].index('loss_u')].numpy())
plt.semilogy(results['val'][:, results['str'].index('loss_b')].numpy())
plt.legend(('objective', 'interor', 'boundary'))
plt.xlabel('steps')
plt.title("convergence")

plt.subplot(2, 2, 4)
plt.scatter(x.detach().numpy(), y.detach().numpy(), c=torch.abs(u.detach().view(-1) - u_true.view(-1)).numpy())
plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=torch.abs(ub.detach().view(-1) - g.view(-1)).numpy())
plt.colorbar()
plt.title("abs. errors")
plt.show()
