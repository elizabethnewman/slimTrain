import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from autoencoder.data import mnist
from autoencoder.training import train_sgd, evaluate
from autoencoder.mnist import SlimTikNetworkMNIST, MNISTAutoencoder


# for reproducibility
torch.manual_seed(20)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': 8}
val_kwargs = {'batch_size': 8}
if use_cuda:
    cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)


# load data
num_train = 2 ** 5
num_val = 2 ** 4
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=num_train, num_val=num_val)

# setup network
net = SlimTikNetworkMNIST()

# loss
criterion = nn.MSELoss(reduction='mean')

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


