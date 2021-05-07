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

# for saving
import shutil
import datetime
import sys

# set seed for reproducibility
torch.manual_seed(20)
torch.set_default_dtype(torch.float32)

# load data
y_train, c_train = get_regression_data(2000)
y_test, c_test = get_regression_data(500)

# create network
feature_extractor = ResidualNetwork(in_features=y_train.shape[1], target_features=c_train.shape[1],
                                    width=8, depth=8, final_time=4, closing_layer=False)
W = torch.randn(c_train.shape[1], 8 + 1)

net = SlimTikNetwork(feature_extractor, W, bias=True, memory_depth=5, lower_bound=1e-7, upper_bound=1e3,
                     opt_method='trial_points', reduction='mean', sumLambda=0.05)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('trainable parameters = ', pytorch_total_params)

# loss function
criterion = nn.MSELoss(reduction='mean')

# optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

# train!
results = train_sgd(net, criterion, optimizer, scheduler, y_train, c_train, y_test, c_test,
                    num_epochs=50, batch_size=10)

# save!
filename = 'tmp'
torch.save((net.state_dict(), results, net.LambdaHist, net.alphaHist), 'results/' + filename + '.pt')
shutil.copy(sys.argv[0], 'results/' + filename + '.pt')

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