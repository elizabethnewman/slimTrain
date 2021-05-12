import os
import shutil
import datetime
import sys
import pickle

import torch
import matplotlib.pyplot as plt

sys.path.append('..')
from networks.resnet import ResidualNetwork
from pinns.data import poisson2D
from pinns.poisson import PoissonPINN
from pinns.training import train_lbfgs
from pinns.utils import set_filename_lbfgs, set_default_arguments_lbfgs


parser = set_default_arguments_lbfgs()
args = parser.parse_args()
print(args)

# for reproducibility
torch.manual_seed(args.seed)
torch.set_default_dtype(torch.float64)

# load data
d = args.input_dim       # domain dimension
n = args.num_interior    # number of interior points
nb = args.num_boundary    # number of boundary points

a = 4.0     # constant
b = 2.0     # constant
data = poisson2D(d, n, nb, a, b)

# build network
feature_extractor = ResidualNetwork(in_features=d, width=args.width, depth=args.depth, final_time=args.final_time,
                                    target_features=1, closing_layer=(not args.no_closing_layer))
pinn = PoissonPINN(feature_extractor)

# optimization
opt = torch.optim.LBFGS(feature_extractor.parameters(), line_search_fn=args.line_search, max_iter=args.max_iter)
results, total_time = train_lbfgs(pinn, opt, data, args.num_epochs)

# plotting
u_true = data['interior_true']
x, y, xb, yb, f, g = pinn.extract_data(data, requires_grad=True)

u = pinn.net_u(torch.cat((x, y), dim=1))
f_pred = pinn.net_f(x, y)
ub = pinn.net_u(torch.cat((xb, yb), dim=1))

sol_loss = torch.norm(u.view(-1) - u_true.view(-1)) / torch.norm(u_true)
lap_loss = torch.norm(f.view(-1) - f_pred.view(-1)) / torch.norm(f)
boundary_loss = torch.norm(ub.view(-1) - g.view(-1))
print('u: %0.4e' % sol_loss)
print('lapu: %0.4e' % lap_loss)
print('boundary: %0.4e' % boundary_loss.item())

# save!
if args.save:
    with torch.no_grad():
        filename, details = set_filename_lbfgs(args)
        stored_results = {'network': pinn, 'optimizer': opt.defaults,
                          'results': results, 'seed': args.seed, 'args': args, 'total_time': total_time,
                          'final_loss': {'sol': sol_loss, 'boundary': boundary_loss, 'lap': lap_loss}}
        if not os.path.exists(args.dirname):
            os.makedirs(args.dirname)
        pickle.dump(stored_results, open(args.dirname + filename + details + '.pt', 'wb'))
        shutil.copy(sys.argv[0], args.dirname + filename + details + '.py')

if args.plot:
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
