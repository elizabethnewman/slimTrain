import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from autoencoder.data import mnist
from autoencoder.mnist_network import MNISTAutoencoder
from autoencoder.training import train_sgd, evaluate
from utils import seed_everything
from autoencoder.utils import set_default_arguments_adam, set_filename_adam

# for saving
import os
import shutil
import sys
import pickle
from datetime import datetime

# parse arguments
parser = set_default_arguments_adam()
args = parser.parse_args()

filename, details = set_filename_adam(args)
now = datetime.now()
my_date = now.strftime("%m-%d-%Y--")
print(my_date + filename + details)
print(args)

# for reproducibility
seed_everything(args.seed)

# setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': args.batch_size}
val_kwargs = {'batch_size': args.test_batch_size}
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
net = MNISTAutoencoder(width_enc=args.width_enc, width_dec=args.width_dec, intrinsic_dim=args.intrinsic_dim).to(device)

# loss
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam([{'params': net.feature_extractor.enc.parameters(), 'weight_decay': args.alpha1},
                        {'params': net.feature_extractor.dec_feature_extractor.parameters(), 'weight_decay': args.alpha1},
                        {'params': net.final_layer.parameters(), 'weight_decay': args.alpha2}],
                       lr=args.lr)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# train!
results, total_time, opt_val_loss_net = train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader,
                                                  device=device, num_epochs=args.num_epochs,
                                                  log_interval=args.log_interval)

# final evaluation of network
train_loss = evaluate(net, criterion, train_loader, device=device)
val_loss = evaluate(net, criterion, val_loader, device=device)
test_loss = evaluate(net, criterion, test_loader, device=device)
final_loss = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
print(final_loss)

# optimal validation loss evaluation of network
opt_val_train_loss = evaluate(opt_val_loss_net, criterion, train_loader, device=device)
opt_val_val_loss = evaluate(opt_val_loss_net, criterion, val_loader, device=device)
opt_val_test_loss = evaluate(opt_val_loss_net, criterion, test_loader, device=device)
opt_val_loss = {'train_loss': opt_val_train_loss, 'val_loss': opt_val_val_loss, 'test_loss': opt_val_test_loss}
print(opt_val_loss)

# save!
if args.save:
    net.to('cpu')
    opt_val_loss_net.to('cpu')

    stored_results = {'network': net, 'opt_val_loss_network': opt_val_loss_net,
                      'final_loss': final_loss, 'opt_val_loss': opt_val_loss,
                      'optimizer': optimizer.defaults, 'scheduler': scheduler.state_dict(),
                      'results': results, 'total_time': total_time}

    if not os.path.exists(args.dirname):
        os.makedirs(args.dirname)

    pickle.dump(stored_results, open(args.dirname + my_date + filename + details + '.pt', 'wb'))
    shutil.copy(sys.argv[0], args.dirname + my_date + filename + details + '.py')
