import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from networks.slimtik import SlimTikNetworkLinearOperatorFull
from slimtik_functions.linear_operators import ConvolutionTranspose2D
from autoencoder.data import mnist
from autoencoder.mnist import MNISTAutoencoderFeatureExtractor
from autoencoder.training import train_sgd, evaluate

# for saving
import os
import shutil
import datetime
import sys
import pickle


# for reproducibility
torch.manual_seed(20)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 8}
val_kwargs = {'batch_size': 8}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)


# load data
num_train = 2 ** 5
num_val = 2 ** 4
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=num_train, num_val=num_val)

# build network
feature_extractor = MNISTAutoencoderFeatureExtractor().to(device)

bias = True
# placeholder for linear operator (empty tensor will be replaced during iterations
linOp = ConvolutionTranspose2D(torch.empty(0, 16, 14, 14), in_channels=16, out_channels=1,
                               kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias)

# W = torch.randn(linOp.numel_in())
# initialize with pytorch
dec2 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                          kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias)
W = dec2.weight.data.reshape(-1)
if bias:
    b = dec2.bias.data.reshape(-1)
    W = torch.cat((W, b))

net = SlimTikNetworkLinearOperatorFull(feature_extractor, linOp, W=W, bias=linOp.bias,
                                       memory_depth=2, lower_bound=1e-7, upper_bound=1e3,
                                       opt_method='trial_points', reduction='sum', sumLambda=5e-2,
                                       total_num_batches=num_train // train_kwargs['batch_size'])

# loss
criterion = nn.MSELoss(reduction='sum')

# optimizer
optimizer = optim.Adam([{'params': net.feature_extractor.enc.parameters(), 'weight_decay': 1e-4},
                        {'params': net.feature_extractor.dec.parameters(), 'weight_decay': 1e-4}], lr=1e-3)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# train!
results, total_time = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device,
                                num_epochs=10, log_interval=1)

# final evaluation of network
train_loss = evaluate(net, criterion, train_loader, device=device)
val_loss = evaluate(net, criterion, val_loader, device=device)
test_loss = evaluate(net, criterion, test_loader, device=device)

# save!
filename = 'autoencoder_mnist_slimtik'
net.to_('cpu')
net.linOp.data = None
stored_results = {'network': net, 'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                  'results': results, 'total_time': total_time,
                  'final_loss': {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}}

if not os.path.exists('results/'):
    os.makedirs('results/')
pickle.dump(stored_results, open('results/' + filename + '.pt', 'wb'))
shutil.copy(sys.argv[0], 'results/' + filename + '.py')


# plot results

with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    outputs = net(inputs).to('cpu')
    inputs = inputs.to('cpu')


plt.figure(1)
n = inputs.shape[0]
m = math.floor(math.sqrt(n))
for i in range(m ** 2):
    plt.subplot(m, m, i + 1)
    plt.imshow(inputs[i].numpy().squeeze())
    plt.colorbar()

plt.title('true (test)')
plt.show()


plt.figure(2)
for i in range(m ** 2):
    plt.subplot(m, m, i + 1)
    plt.imshow(outputs[i].numpy().squeeze())
    plt.title('%0.2e' % ((torch.norm(outputs[i] - inputs[i]) / torch.norm(inputs[i])).item()))
    plt.colorbar()

plt.show()


plt.figure(3)
plt.semilogy(results['val'][:, results['str'].index('train_loss')].numpy())
plt.semilogy(results['val'][:, results['str'].index('val_loss')].numpy())
plt.legend(('training', 'validation'))
plt.xlabel('epochs')
plt.title("convergence")
plt.show()
