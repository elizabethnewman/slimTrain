import os
import sys
import shutil
from datetime import datetime
import pickle


import torch
import matplotlib.pyplot as plt

sys.path.append('..')
from networks.resnet import ResidualNetwork
from pinns.data import poisson2D
from pinns.poisson import PoissonPINNSlimTik
from pinns.training import train_sgd
from pinns.utils import set_filename_slimtik, set_default_arguments_slimtik


parser = set_default_arguments_slimtik()
args = parser.parse_args()
print(args)

# for reproducibility
torch.manual_seed(args.seed)
torch.set_default_dtype(torch.float64)

# load data
d = args.input_dim      # domain dimension
n = args.num_interior    # number of interior points
nb = args.num_boundary    # number of boundary points

a = 4.0     # constant
b = 2.0     # constant
data = poisson2D(d, n, nb, a, b)

# build network
feature_extractor = ResidualNetwork(in_features=d, width=args.width, depth=args.depth, final_time=args.final_time,
                                    target_features=1, closing_layer=False)

# W = torch.randn(1, args.width + bias=(not args.no_bias))
# initialize using pytorch
tmp_layer = torch.nn.Linear(args.width, 1, bias=(not args.no_bias))
W = tmp_layer.weight.data
b = tmp_layer.bias.data
W = torch.cat((W, b.unsqueeze(1)), dim=1)


pinn = PoissonPINNSlimTik(feature_extractor, W, bias=(not args.no_bias), memory_depth=args.mem_depth,
                          lower_bound=args.lower_bound, upper_bound=args.upper_bound,
                          opt_method=args.opt_method, reduction=args.reduction, sumLambda=args.sum_lambda)

# optimization
opt = torch.optim.Adam(feature_extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
results, total_time = train_sgd(pinn, opt, scheduler, data, num_epochs=args.num_epochs, batch_size=args.batch_size,
                                log_interval=args.log_interval)

# final results
u_true = data['interior_true']
x, y, xb, yb, f, g = pinn.extract_data(data, requires_grad=True)

u = pinn.net_u(torch.cat((x, y), dim=1))
f_pred = pinn.net_f(x, y)
ub = pinn.net_u(torch.cat((xb, yb), dim=1))

loss_u = (torch.norm(u.view(-1) - u_true.view(-1)) / torch.norm(u_true)).item()
loss_lapu = (torch.norm(f.view(-1) - f_pred.view(-1)) / torch.norm(f)).item()
loss_b = (torch.norm(ub.view(-1) - g.view(-1))).item()
print('u: %0.4e' % loss_u)
print('lapu: %0.4e' % loss_lapu)
print('boundary: %0.4e' % loss_b)

# save!
if args.save:
    with torch.no_grad():
        filename, details = set_filename_slimtik(args)
        stored_results = {'network': pinn, 'optimizer': opt.defaults, 'scheduler': scheduler.state_dict(),
                          'results': results, 'total_time': total_time,
                          'final_loss': {'loss_u': loss_u, 'loss_lapu': loss_lapu, 'loss_b': loss_b}}
        if not os.path.exists(args.dirname):
            os.makedirs(args.dirname)

        now = datetime.now()
        my_date = now.strftime("%m-%d-%Y--")
        pickle.dump(stored_results, open(args.dirname + my_date + filename  + details + '.pt', 'wb'))
        shutil.copy(sys.argv[0], args.dirname + my_date + filename + details + '.py')

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
    plt.figure(3)
    plt.subplot(2, 2, 1)
    plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=g.detach().numpy())
    plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u_true.numpy())

    plt.colorbar()
    plt.title("utrue")

    plt.subplot(2, 2, 2)
    plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=ub.detach().view(-1).numpy())
    plt.scatter(x.detach().numpy(), y.detach().numpy(), c=u.detach().view(-1).numpy())
    plt.colorbar()
    plt.title("approx")

    plt.subplot(2, 2, 3)
    plt.semilogy(results['val'][:, results['str'].index('loss')].numpy())
    plt.semilogy(results['val'][:, results['str'].index('loss_u')].numpy())
    plt.semilogy(results['val'][:, results['str'].index('loss_b')].numpy())
    plt.legend(('objective', 'interior', 'boundary'))
    plt.xlabel('steps')
    plt.title("convergence")

    plt.subplot(2, 2, 4)
    plt.scatter(xb.detach().numpy(), yb.detach().numpy(), c=torch.abs(ub.detach().view(-1) - g.view(-1)).numpy())
    plt.scatter(x.detach().numpy(), y.detach().numpy(), c=torch.abs(u.detach().view(-1) - u_true.view(-1)).numpy())
    plt.colorbar()
    plt.title("abs. errors")
    plt.show()
