import torch
import argparse


def argument_parser():
    # training settings
    parser = argparse.ArgumentParser(description='pytorch-slimtik')

    # reproducibility options
    parser.add_argument('--seed', type=int, default=1, metavar='seed',
                        help='random seed (default: 1)')

    # gpu
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='use tensor network')

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

    # ADAM optimization options
    parser.add_argument('--num-epochs', type=int, default=50, metavar='num_epochs',
                        help='maximum of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='weight_decay',
                        help='regularization on network weights (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='batch_size',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='test_batch_size',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--betas', type=int, default=(0.9, 0.99), metavar='betas', nargs='+',
                        help='Adam betas (default: (0.9, 0.99))')

    # scheduler options
    parser.add_argument('--gamma', type=float, default=1, metavar='gamma',
                        help='scheduler learning rate reduction gamma (default: 1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='step_size',
                        help='scheduler learning rate reduction step size (default: 1)')

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

    # saving options (printouts in terminal will not display
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the current model')
    parser.add_argument('--filename', type=str, default='tmp',
                        help='file name to save model')
    parser.add_argument('--dirname', type=str, default='results/',
                        help='file name to save model - including "/"')

    # printing/plotting options
    parser.add_argument('--plot', action='store_true', default=False,
                        help='show plots')
    parser.add_argument('--no-verbose', action='store_true', default=False,
                        help='turn off verbosity')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser


def seed_everything(seed):
    # option to add numpy, random, etc.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# optimizer printouts
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
                frmt += optimizer_params_format(p, spacing=spacing, precision=precision)

        else:
            names += (name,)
            val += [param]
            frmt += optimizer_params_format(param, spacing=spacing, precision=precision)

    opt_params = {'str': names, 'frmt': frmt, 'val': val}

    return opt_params


def optimizer_params_format(param, spacing=15, precision=4):
    if isinstance(param, int):
        frmt = '{:<' + str(spacing) + 'd}'
    else:
        frmt = '{:<' + str(spacing) + '.' + str(precision) + 'e}'

    return frmt


# extract and insert data
def module_getattr(obj, names):
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return getattr(obj, names)
    else:
        return module_getattr(getattr(obj, names[0]), names[1:])


def module_setattr(obj, names, val):
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return setattr(obj, names, val)
    else:
        return module_setattr(getattr(obj, names[0]), names[1:], val)


def extract_data(net, attr: str='data'):
    theta = torch.empty(0)
    for name, w in net.named_parameters():
        if getattr(w, attr) is None:
            w = torch.zeros_like(w.data)
        else:
            w = getattr(w, attr)

        theta = torch.cat((theta.to(w.device), w.reshape(-1)))

    return theta


def insert_data(net, theta):

    count = 0
    for name, w in net.named_parameters():
        name_split = name.split('.')
        n = w.numel()
        module_setattr(net, name_split + ['data'], theta[count:count + n].reshape(w.shape))
        count += n
