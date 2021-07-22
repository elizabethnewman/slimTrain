
from utils import argument_parser


def add_default_arguments():
    parser = argument_parser()
    parser.add_argument('--intrinsic-dim', type=int, default=50, metavar='intrinsic_dim',
                        help='intrinsic dimension of autoencoder (default: 50)')
    parser.add_argument('--width-enc', type=int, default=16, metavar='width_enc',
                        help='width of encoder (default: 16)')
    parser.add_argument('--width-dec', type=int, default=16, metavar='width_dec',
                        help='width of decoder (default: 16)')
    parser.add_argument('--save-intermediate', action='store_true', default=False,
                        help='save intermediate training images (default: False)')
    parser.add_argument('--alpha1', type=int, default=1e-4, metavar='alpha1',
                        help='regularization on first block of weights default: 1e-10)')
    parser.add_argument('--alpha2', type=int, default=1e-4, metavar='alpha2',
                        help='regularization on final block of weights default: 1e-10)')

    return parser


def set_default_arguments_adam():
    parser = add_default_arguments()

    parser.set_defaults(seed=20,
                        width_enc=16, width_dec=16, depth=8, final_time=4,
                        num_train=2**10, num_val=2**5,
                        reduction='sum',
                        num_epochs=100, batch_size=32, lr=1e-3, weight_decay=0,
                        step_size=25, gamma=1,
                        dirname='results/autoencoder/')

    return parser


def set_filename_adam(args):
    filename = 'autoencoder_mnist_adam'
    details = "--width-enc-%0.2d--width-dec-%0.2d--num-train-%d--lr-%0.2e--alpha1-%0.2e--alpha2-%0.2e" \
              % (args.width_enc, args.width_dec, args.num_train, args.lr, args.alpha1, args.alpha2)
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
