
# for saving
import os
import shutil
from datetime import datetime
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math

sys.path.append('..')
from autoencoder.data import mnist
from autoencoder.mnist import MNISTAutoencoderSlimTik
from autoencoder.training import train_sgd, evaluate
from autoencoder.utils import set_filename_slimtik, set_default_arguments_slimtik

parser = set_default_arguments_slimtik()
args = parser.parse_args()

# for reproducibility
torch.manual_seed(args.seed)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': args.batch_size}
val_kwargs = {'batch_size': 32}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)


# load data
train_loader, val_loader, test_loader = mnist(train_kwargs, val_kwargs, num_train=args.num_train, num_val=args.num_val,
                                              dirname='../autoencoder/')

# build network
net = MNISTAutoencoderSlimTik(width=args.width, intrinsic_dim=args.intrinsic_dim, bias=True,
                              memory_depth=args.mem_depth, lower_bound=args.lower_bound, upper_bound=args.upper_bound,
                              opt_method=args.opt_method,
                              reduction=args.reduction, sumLambda=args.sum_lambda,
                              device=device).to(device)

# loss
criterion = nn.MSELoss(reduction=args.reduction)

# optimizer
optimizer = optim.Adam([{'params': net.feature_extractor.enc.parameters(), 'weight_decay': args.weight_decay},
                        {'params': net.feature_extractor.dec.parameters(), 'weight_decay': args.weight_decay}],
                       lr=args.lr)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# train!
results, total_time = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader, device=device,
                                num_epochs=args.num_epochs, log_interval=args.log_interval)

# final evaluation of network
train_loss = evaluate(net, criterion, train_loader, device=device)
val_loss = evaluate(net, criterion, val_loader, device=device)
test_loss = evaluate(net, criterion, test_loader, device=device)

# save!
if args.save:
    filename, details = set_filename_slimtik(args)
    net.clear_()
    net.to_('cpu')
    stored_results = {'network': net, 'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                      'results': results, 'total_time': total_time,
                      'final_loss': {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}}
    if not os.path.exists(args.dirname):
        os.makedirs(args.dirname)

    now = datetime.now()
    my_date = now.strftime("%m-%d-%Y--")
    pickle.dump(stored_results, open(args.dirname + my_date + filename + details + '.pt', 'wb'))
    shutil.copy(sys.argv[0], args.dirname + my_date + filename + details + '.py')

# plot results
if args.plot:
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
