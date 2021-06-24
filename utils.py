import torch
import argparse


def argument_parser():
    # training settings
    parser = argparse.ArgumentParser(description='pytorch-slimtik')

    # resnet options
    parser.add_argument('--width', type=int, default=4, metavar='width',
                        help='width of network (default: 4)')
    parser.add_argument('--depth', type=int, default=4, metavar='depth',
                        help='depth of network (default: 4)')
    parser.add_argument('--final-time', type=float, default=1, metavar='final_time',
                        help='final time corresponding to network depth (default: 1)')
    parser.add_argument('--no-closing-layer', action='store_true', default=False,
                        help='final linear layer as part of resnet')

    # slimtik options
    parser.add_argument('--mem-depth', type=int, default=0, metavar='mem_depth',
                        help='slimtik memory depth (default: 0)')
    parser.add_argument('--lower-bound', type=float, default=1e-7, metavar='lower_bound',
                        help='slimtik lower bound (default: 1e-7)')
    parser.add_argument('--upper-bound', type=float, default=1e3, metavar='upper_bound',
                        help='slimtik upper bound (default: 1e3)')
    parser.add_argument('--opt-method', type=str, default='none', metavar='opt_method',
                        help='slimtik optimization method, either "none" or "trial_points" (default: "none")')
    parser.add_argument('--sum-lambda', type=float, default=0.05, metavar='sum_lambda',
                        help='slimtik sum of lambda initialization (default: 0.05)')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='slimtik no bias option (default: False)')

    # data options
    parser.add_argument('--num-train', type=int, default=1000, metavar='num_train',
                        help='number of training samples (default: 1000)')
    parser.add_argument('--num-val', type=int, default=100, metavar='num_val',
                        help='number of validation samples (default: 100)')
    parser.add_argument('--num-test', type=int, default=100, metavar='num_test',
                        help='number of testing samples (default: 100)')

    # loss options
    parser.add_argument('--reduction', type=str, default='mean', metavar='reduction',
                        help='type of reduction in loss function (default: "mean")')

    # optimization options
    parser.add_argument('--num-epochs', type=int, default=5, metavar='num_epochs',
                        help='maximum of epochs to train (default: 5)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='batch_size',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='test_batch_size',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, metavar='weight_decay',
                        help='regularization on network weights (default: 0.01)')
    parser.add_argument('--betas', type=int, default=(0.9, 0.99), metavar='betas', nargs='+',
                        help='Adam betas (default: (0.9, 0.99))')
    parser.add_argument('--max-iter', type=int, default=10, metavar='max_iter',
                        help='maximum number of internal L-BFGS iterations (default: 10)')
    parser.add_argument('--line-search', type=str, default='strong_wolfe', metavar='line_search',
                        help='line search method for L-BFGS iterations (default: "strong_wolfe")')

    # scheduler
    parser.add_argument('--gamma', type=float, default=1, metavar='gamma',
                        help='scheduler learning rate reduction gamma (default: 1)')
    parser.add_argument('--step-size', type=int, default=1, metavar='step_size',
                        help='scheduler learning rate reduction step size (default: 1)')

    # reproducibility options
    parser.add_argument('--seed', type=int, default=1, metavar='seed',
                        help='random seed (default: 1)')

    # saving options (printouts in terminal will not display
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the current model')
    parser.add_argument('--filename', type=str, default='tmp',
                        help='file name to save model')
    parser.add_argument('--dirname', type=str, default='results/',
                        help='file name to save model - including "/"')

    # gpu
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='use tensor network')

    # printing/plotting options
    parser.add_argument('--plot', action='store_true', default=False,
                        help='show plots')
    parser.add_argument('--no-verbose', action='store_true', default=False,
                        help='turn off verbosity')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser


def extract_parameters(net):
    with torch.no_grad():
        out = torch.empty(0)
        for p in net.parameters():
            out = torch.cat((out, p.data.reshape(-1).to('cpu')))

    return out


def extract_gradients(net):
    out = torch.empty(0)
    for p in net.parameters():
        out = torch.cat((out, p.grad.reshape(-1).to('cpu')))

    return out


def parameters_norm(net):
    with torch.no_grad():
        out = None
        for p in net.parameters():
            if out is None:
                out = torch.sum(p.data.to('cpu') ** 2)
            else:
                out += torch.sum(p.data.to('cpu') ** 2)

    return torch.sqrt(out).item()


def grad_norm(net: torch.nn.Module):

    out = torch.zeros(1)
    for p in net.parameters():
        if p.grad is None:
            continue

        out += torch.sum(p.grad.to('cpu') ** 2)

    return torch.sqrt(out).item()


def optimizer_params(optimizer, spacing=15, precision=4):
    keys = list(optimizer.param_groups[0].keys())
    keys = tuple(keys[1:])

    names = ()
    val = []
    frmt = ''
    for name in keys:
        param = optimizer.param_groups[0][name]
        if isinstance(param, str) or isinstance(param, bool):
            continue
        elif isinstance(param, tuple):
            names += len(param) * (name,)
            val += [*param]
            for p in param:
                frmt += parameter_format(p, spacing=spacing, precision=precision)

        else:
            names += (name,)
            val += [param]
            frmt += parameter_format(param, spacing=spacing, precision=precision)

    opt_params = {'str': names, 'frmt': frmt, 'val': val}

    return opt_params


def parameter_format(param, spacing=15, precision=4):

    if isinstance(param, int):
        frmt = '{:<' + str(spacing) + 'd}'
    else:
        frmt = '{:<' + str(spacing) + '.' + str(precision) + 'e}'

    return frmt

