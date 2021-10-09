
from utils import argument_parser


def add_default_arguments():
    parser = argument_parser()
    parser.add_argument('--intrinsic-dim', type=int, default=50, metavar='intrinsic_dim',
                        help='intrinsic dimension of autoencoder (default: 50)')
    parser.add_argument('--width-enc', type=int, default=16, metavar='width_enc',
                        help='width of encoder (default: 16)')
    parser.add_argument('--width-dec', type=int, default=16, metavar='width_dec',
                        help='width of decoder (default: 16)')
    parser.add_argument('--alpha1', type=float, default=1e-4, metavar='alpha1',
                        help='regularization on first block of weights default: 1e-10)')
    parser.add_argument('--alpha2', type=float, default=1e-4, metavar='alpha2',
                        help='regularization on final block of weights default: 1e-10)')

    return parser


def set_default_arguments_adam():
    parser = add_default_arguments()

    parser.set_defaults(seed=20,
                        width_enc=16, width_dec=16,
                        num_train=50000, num_val=10000,
                        num_epochs=100, batch_size=32, lr=1e-3,
                        reduction='sum',
                        dirname='results/autoencoder/')

    return parser


def set_filename_adam(args):
    filename = 'autoencoder_mnist_adam'
    details = ("--epochs-%d--num-train-%d--alpha1-%0.2e--alpha2-%0.2e--seed-%d"
               % (args.num_epochs, args.num_train, args.alpha1, args.alpha2, args.seed))
    return filename, details


def set_default_arguments_slimtrain():
    parser = set_default_arguments_adam()
    parser.set_defaults(mem_depth=0, sum_lambda=5e-2, opt_method='trial_points',
                        lower_bound=1e-7, upper_bound=1e1)
    return parser


def set_filename_slimtrain(args):
    filename = 'autoencoder_mnist_slimtrain'

    _, details = set_filename_adam(args)
    details += ("--mem-depth-%0.2d" % args.mem_depth)
    return filename, details
