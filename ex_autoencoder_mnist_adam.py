import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from autoencoder.data import mnist
from autoencoder.mnist_network import MNISTAutoencoder
from autoencoder.training import train_sgd, evaluate
from utils import seed_everything

# for saving
import os
import shutil
import sys
import pickle


# for reproducibility
seed_everything(20)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 16}
val_kwargs = {'batch_size': 16}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)


# load data
num_train = 2 ** 5
num_val = 2 ** 4
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=num_train, num_val=num_val,
                                              dirname='autoencoder/')

# build network
net = MNISTAutoencoder().to(device)

# loss
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam([{'params': net.feature_extractor.enc.parameters(), 'weight_decay': 1e-5},
                        {'params': net.feature_extractor.dec_feature_extractor.parameters(), 'weight_decay': 1e-5},
                        {'params': net.final_layer.parameters(), 'weight_decay': 1e-5}],
                       lr=1e-3)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=25, gamma=1)

# train!
results, total_time, opt_val_loss_net = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader,
                                                  device=device, num_epochs=10, log_interval=1)

# final evaluation of network
train_loss = evaluate(net, criterion, train_loader, device=device)
val_loss = evaluate(net, criterion, val_loader, device=device)
test_loss = evaluate(net, criterion, test_loader, device=device)
final_loss = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}

# optimal validation loss evaluation of network
opt_val_train_loss = evaluate(opt_val_loss_net, criterion, train_loader, device=device)
opt_val_val_loss = evaluate(opt_val_loss_net, criterion, val_loader, device=device)
opt_val_test_loss = evaluate(opt_val_loss_net, criterion, test_loader, device=device)
opt_val_loss = {'train_loss': opt_val_train_loss, 'val_loss': opt_val_val_loss, 'test_loss': opt_val_test_loss}

# save!
filename = 'autoencoder_mnist_adam'
net.to('cpu')
opt_val_loss_net.to('cpu')
stored_results = {'network': net.to('cpu'), 'opt_val_loss_network': opt_val_loss_net,
                  'final_loss': final_loss, 'opt_val_loss': opt_val_loss,
                  'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                  'results': results, 'total_time': total_time}

if not os.path.exists('results/'):
    os.makedirs('results/')
pickle.dump(stored_results, open('results/' + filename + '.pt', 'wb'))
shutil.copy(sys.argv[0], 'results/' + filename + '.py')

#%% ================================================================================================================ %%#
# Plot results

with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    outputs = net(inputs).to('cpu')
    inputs = inputs.to('cpu')


plt.figure(1)
n = 9
m = math.floor(math.sqrt(n))
for i in range(m ** 2):
    plt.subplot(m, m, i + 1)
    plt.imshow(inputs[i].numpy().squeeze())
    plt.axis('off')
    plt.colorbar()

plt.title('true (test)')
plt.show()


plt.figure(2)
for i in range(m ** 2):
    plt.subplot(m, m, i + 1)
    plt.imshow(outputs[i].numpy().squeeze())
    plt.axis('off')
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
