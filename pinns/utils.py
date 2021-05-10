
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
                        dirname='results/')

    return parser


def set_filename_lbfgs(args):
    filename = 'pinns_poisson_lbfgs'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e" \
              % (args.width, args.depth, args.lr, args.weight_decay)
    return filename, details

