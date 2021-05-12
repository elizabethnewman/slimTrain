
from utils import argument_parser


def add_default_arguments():
    parser = argument_parser()
    parser.add_argument('--intrinsic-dim', type=int, default=50, metavar='intrinsic_dim',
                        help='intrinsic dimension of autoencoder (default: 50)')

    return parser


def set_default_arguments_adam():
    parser = add_default_arguments()

    parser.set_defaults(seed=20,
                        width=16, depth=8, final_time=4,
                        num_train=2**10, num_val=2**5,
                        reduction='sum',
                        num_epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-4,
                        step_size=25, gamma=1,
                        dirname='results/autoencoder/')

    return parser


def set_filename_adam(args):
    filename = 'autoencoder_mnist_adam'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e" \
              % (args.width, args.depth, args.lr, args.weight_decay)
    return filename, details


def set_default_arguments_slimtik():
    parser = set_default_arguments_adam()
    parser.set_defaults(mem_depth=0, sum_lambda=5e-2, opt_method='none')
    return parser


def set_filename_slimtik(args):
    filename = 'autoencoder_mnist_slimtik'

    _, details = set_filename_adam(args)
    details += "--memdepth-%0.2d--optmethod-%s" % (args.mem_depth, args.opt_method)
    return filename, details