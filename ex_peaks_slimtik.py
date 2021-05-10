import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from networks.resnet import ResidualNetwork
from networks.slimtik import SlimTikNetwork
from peaks.data import get_regression_data, visualize_regression_image
from peaks.training import train_sgd
import pickle
from peaks.utils import set_default_arguments_slimtik, set_filename_slimtik

# for saving
import shutil
import datetime
import sys

parser = set_default_arguments_slimtik()
args = parser.parse_args()
print(args)


# set seed for reproducibility
torch.manual_seed(args.seed)
# torch.set_default_dtype(torch.float32)

# load data
y_train, c_train = get_regression_data(args.num_train)
y_test, c_test = get_regression_data(args.num_test)

# create network
feature_extractor = ResidualNetwork(in_features=y_train.shape[1], target_features=c_train.shape[1],
                                    width=args.width, depth=args.depth, final_time=args.final_time,
                                    closing_layer=(not args.no_closing_layer))
W = torch.randn(c_train.shape[1], args.width + int(not args.no_bias))


net = SlimTikNetwork(feature_extractor, W, bias=(not args.no_bias), memory_depth=args.mem_depth,
                     lower_bound=args.lower_bound, upper_bound=args.upper_bound,
                     opt_method=args.opt_method, reduction=args.reduction, sumLambda=args.sum_lambda)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('trainable parameters = ', pytorch_total_params)

# loss function
criterion = nn.MSELoss(reduction=args.reduction)

# optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# train!
results = train_sgd(net, criterion, optimizer, scheduler, y_train, c_train, y_test, c_test,
                    num_epochs=args.max_epochs, batch_size=args.batch_size)

# save!
if args.save:
    with torch.no_grad():
        filename, details = set_filename_slimtik(args)
        stored_results = {'network': net, 'optimizer': optimizer, 'scheduler': scheduler, 'criterion': criterion,
                          'results': results, 'seed': args.seed, 'args': args}
        pickle.dump(stored_results, open(args.dirname + filename + details + '.pt', 'wb'))
        shutil.copy(sys.argv[0], args.dirname + filename + details + '.py')


if args.plot:
    # plot results
    plt.figure(1)
    plt.semilogy(results['val'][:, results['str'].index('loss')].numpy(), 'b')
    plt.semilogy(results['val'][:, results['str'].index('loss') + 2].numpy(), 'r--')
    plt.legend(('training', 'validation'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("convergence")
    plt.show()

    plt.figure(2)
    visualize_regression_image(net)
    plt.show()

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.semilogy(net.alphaHist, 'o')
    plt.xlabel('iteration')
    plt.ylabel('alpha = sumLambda / iter')

    plt.subplot(2, 1, 2)
    lambdaTmp = np.array(net.LambdaHist)
    idx = np.arange(len(net.LambdaHist))
    plt.semilogy(idx[lambdaTmp >= 0], lambdaTmp[lambdaTmp >= 0], 'bo')
    plt.semilogy(idx[lambdaTmp < 0], -lambdaTmp[lambdaTmp < 0], 'rx')
    plt.xlabel('iteration')
    plt.ylabel('Lambda')
    plt.legend(('positive', 'negative'))
    plt.show()