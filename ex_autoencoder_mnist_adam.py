import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from autoencoder.data import mnist
from autoencoder.mnist import MNISTAutoencoder
from autoencoder.training import train_sgd, evaluate



# setup
use_cuda = torch.cuda.is_available()
torch.manual_seed(20)
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
net = MNISTAutoencoder()

# loss
criterion = nn.MSELoss(reduction='sum')

# optimizer
optimizer = optim.Adam([{'params': net.enc.parameters(), 'weight_decay': 1e-4},
                        {'params': net.dec.parameters(), 'weight_decay': 1e-4}], lr=1e-3)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=25, gamma=1)

# train!
results = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device, num_epochs=50)
test_results = evaluate(net, criterion, test_loader, device=device)

# save!
# filename = 'tmp'
# torch.save((net.state_dict(), results), 'results/' + filename + '.pt')
# shutil.copy(sys.argv[0], 'results/' + filename + '.pt')

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