
from utils import argument_parser


def add_default_arguments():
    parser = argument_parser()
    parser.add_argument('--input-dim', type=int, default=2, metavar='input_dim',
                        help='dimension of domain (default: 2)')
    parser.add_argument('--num-interior', type=int, default=1000, metavar='num_interior',
                        help='number of interior data points (default: 1000)')
    parser.add_argument('--num-boundary', type=int, default=100, metavar='num_boundary',
                        help='number of boundary data points (default: 100)')

    return parser


def set_default_arguments_lbfgs():
    parser = add_default_arguments()

    parser.set_defaults(seed=20,
                        width=10, depth=10, final_time=10, no_closing_layer=False,
                        max_iter=50, line_search='strong_wolfe',
                        max_epochs=10,
                        dirname='results/pinns/')

    return parser


def set_filename_lbfgs(args):
    filename = 'pinns_poisson_lbfgs'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e" \
              % (args.width, args.depth, args.lr, args.weight_decay)
    return filename, details


def set_default_arguments_adam():
    parser = set_default_arguments_lbfgs()

    parser.set_defaults(max_epochs=100, batch_size=100, log_interval=10,
                        lr=1e-2, weight_decay=1e-2,
                        step_size=500, gamma=0.5)

    return parser


def set_filename_adam(args):
    filename = 'pinns_poisson_adam'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e" \
              % (args.width, args.depth, args.lr, args.weight_decay)
    return filename, details


def set_default_arguments_slimtik():
    parser = set_default_arguments_adam()

    parser.set_defaults(no_closing_layer=True)

    return parser


def set_filename_slimtik(args):
    filename = 'pinns_poisson_slimtik'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e--memdepth-%0.2d--optmethod--%s" \
              % (args.width, args.depth, args.lr, args.weight_decay, args.mem_depth, args.opt_method)
    return filename, details
