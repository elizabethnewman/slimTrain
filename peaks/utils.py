
from utils import argument_parser


def set_default_arguments_adam():
    parser = argument_parser()

    parser.set_defaults(seed=20,
                        width=8, depth=8, final_time=4, no_closing_layer=False,
                        num_train=2000, num_test=500,
                        reduction='mean',
                        max_epochs=50, batch_size=10, lr=1e-3, weight_decay=1e-4,
                        step_size=25, gamma=0.5,
                        dirname='results/')

    return parser


def set_filename_adam(args):
    filename = 'peaks_adam'
    details = "--width-%0.2d--depth-%0.2d--lr-%0.2e--decay-%0.2e" \
              % (args.width, args.depth, args.lr, args.weight_decay)
    return filename, details


def set_default_arguments_slimtik():
    parser = set_default_arguments_adam()
    parser.set_defaults(no_closing_layer=True)
    return parser


def set_filename_slimtik(args):
    filename = 'peaks_slimtik'

    _, details = set_filename_adam(args)
    details += "--memdepth-%0.2d--optmethod-%s" % (args.mem_depth, args.opt_method)
    return filename, details

