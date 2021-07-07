import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from autoencoder.data import mnist
from autoencoder.mnist import MNISTAutoencoderSlimTik
from autoencoder.training import train_sgd, evaluate


import cProfile
import profile

# for saving
import os
import shutil
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
num_train = 2 ** 10
num_val = 2 ** 5
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=num_train, num_val=num_val)

# build network
net = MNISTAutoencoderSlimTik(memory_depth=2, opt_method='trial_points', device=device).to(device)

# loss
criterion = nn.MSELoss(reduction='mean')

# optimizer
optimizer = optim.Adam([{'params': net.feature_extractor.enc.parameters(), 'weight_decay': 1e-4},
                        {'params': net.feature_extractor.dec.parameters(), 'weight_decay': 1e-4}], lr=1e-3)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# # train!
results, total_time = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device,
                                num_epochs=2, log_interval=1)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# cProfile.run('train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device,num_epochs=2, log_interval=1)', sort='tottime')
# profile.run('train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device,num_epochs=2, log_interval=1)', sort='tottime')

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ]
# ) as p:
#     train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device, num_epochs=2, log_interval=1)
# print(p.key_averages().table(
#     sort_by="self_cuda_time_total", row_limit=-1))

# final evaluation of network
train_loss = evaluate(net, criterion, train_loader, device=device)
val_loss = evaluate(net, criterion, val_loader, device=device)
test_loss = evaluate(net, criterion, test_loader, device=device)

# save!
filename = 'autoencoder_mnist_slimtik'
net.to_('cpu')
stored_results = {'network': net, 'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                  'results': results, 'total_time': total_time,
                  'final_loss': {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}}

if not os.path.exists('results/'):
    os.makedirs('results/')
pickle.dump(stored_results, open('results/' + filename + '.pt', 'wb'))
shutil.copy(sys.argv[0], 'results/' + filename + '.py')


# plot results
net.eval()
with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    outputs = net(inputs, inputs).to('cpu')
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
