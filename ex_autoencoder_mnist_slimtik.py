import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from networks.slimtik import SlimTikNetworkLinearOperator
from slimtik_functions.linear_operators import ConvolutionTranspose2D
from autoencoder.data import mnist
from autoencoder.mnist import MNISTAutoencoderFeatureExtractor
from autoencoder.training import train_sgd, evaluate

# for saving
import shutil
import datetime
import sys
import pickle


# for reproducibility
torch.manual_seed(20)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 32}
val_kwargs = {'batch_size': 32}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)


# load data
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=2**10, num_val=2**5)

# build network
feature_extractor = MNISTAutoencoderFeatureExtractor()

# placeholder for linear operator (empty tensor will be replaced during iterations
linOp = ConvolutionTranspose2D(torch.empty(0, 16, 14, 14), in_channels=16, out_channels=1,
                               kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

# W = torch.randn(linOp.numel_in())
# initialize with pytorch
dec2 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                          kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
W = dec2.weight.data.reshape(-1)
b = dec2.bias.data.reshape(-1)
W = torch.cat((W, b))

net = SlimTikNetworkLinearOperator(feature_extractor, linOp, W=W, bias=linOp.bias,
                                   memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                                   opt_method=None, reduction='sum', sumLambda=5e-2)

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
stored_results = {'network': net, 'optimizer': optimizer, 'scheduler': scheduler,
                  'results': results, 'total_time': total_time,
                  'final_loss': {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}}
pickle.dump(stored_results, open('results/' + filename + '.pt', 'wb'))
shutil.copy(sys.argv[0], 'results/' + filename + '.py')


# plot results

with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    outputs = net(inputs)


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
