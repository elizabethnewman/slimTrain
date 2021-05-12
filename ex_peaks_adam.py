import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from networks.resnet import ResidualNetwork
from peaks.data import get_regression_data, visualize_regression_image
from peaks.training import train_sgd, evaluate

# for saving
import os
import shutil
import datetime
import sys
import pickle

# set seed for reproducibility
torch.manual_seed(20)
torch.set_default_dtype(torch.float32)

# load data
y_train, c_train = get_regression_data(2000)
y_val, c_val = get_regression_data(500)
y_test, c_test = get_regression_data(100)

# create network
net = ResidualNetwork(in_features=y_train.shape[1], target_features=c_train.shape[1],
                      width=8, depth=8, final_time=4, closing_layer=True)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('trainable parameters = ', pytorch_total_params)

# loss function
criterion = nn.MSELoss(reduction='mean')

# optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

# train!
results, total_time = train_sgd(net, criterion, optimizer, scheduler, y_train, c_train, y_val, c_val,
                                num_epochs=20, batch_size=10)

# overall loss after training
train_loss, _ = evaluate(net, criterion, y_train, c_train)
val_loss, _ = evaluate(net, criterion, y_val, c_val)
test_loss, _ = evaluate(net, criterion, y_test, c_test)
print('final loss - train: %0.4e\tval: %0.4e\ttest: %0.4e' % (train_loss, val_loss, test_loss))

# save!
filename = 'peaks_adam'
stored_results = {'network': net, 'optimizer': optimizer, 'scheduler': scheduler, 'criterion': criterion,
                  'results': results, 'total_time': total_time,
                  'final_loss': {'train': train_loss, 'val': val_loss, 'test': test_loss}}

if not os.path.exists('results/'):
    os.makedirs('results/')s
pickle.dump(stored_results, open('results/' + filename + '.pt', 'wb'))
shutil.copy(sys.argv[0], 'results/' + filename + '.py')

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