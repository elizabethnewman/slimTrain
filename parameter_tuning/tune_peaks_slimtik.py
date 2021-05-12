import os
import sys
import shutil
import datetime
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from networks.resnet import ResidualNetwork
from networks.slimtik import SlimTikNetwork
from peaks.data import get_regression_data, visualize_regression_image
from peaks.training import train_sgd, evaluate
from peaks.utils import set_default_arguments_slimtik, set_filename_slimtik


parser = set_default_arguments_slimtik()
args = parser.parse_args()
print(args)


# set seed for reproducibility
torch.manual_seed(args.seed)
# torch.set_default_dtype(torch.float32)

# load data
y_train, c_train = get_regression_data(args.num_train)
y_val, c_val = get_regression_data(args.num_val)
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
results, total_time = train_sgd(net, criterion, optimizer, scheduler, y_train, c_train, y_val, c_val,
                                num_epochs=args.num_epochs, batch_size=args.batch_size)

# overall loss after training
train_loss, _ = evaluate(net, criterion, y_train, c_train)
val_loss, _ = evaluate(net, criterion, y_val, c_val)
test_loss, _ = evaluate(net, criterion, y_test, c_test)
print('final loss - train: %0.4e\tval: %0.4e\ttest: %0.4e' % (train_loss, val_loss, test_loss))

# save!
if args.save:
    with torch.no_grad():
        filename, details = set_filename_slimtik(args)
        stored_results = {'network': net, 'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                          'criterion': criterion, 'seed': args.seed, 'args': args,
                          'results': results, 'total_time': total_time,
                          'final_loss': {'train': train_loss, 'val': val_loss, 'test': test_loss}}
        if not os.path.exists(args.dirname):
            os.makedirs(args.dirname)
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