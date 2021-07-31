import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import math
from autoencoder.data import mnist
from autoencoder.mnist2 import MNISTAutoencoderSlimTik
from autoencoder.training import train_sgd, evaluate


torch.set_default_dtype(torch.float64)

# for reproducibility
torch.manual_seed(20)

# load data
num_train = 2 ** 10
batch_size = 8
num_batch = num_train // batch_size
train_data = torch.randn(num_train, 1, 28, 28)


lambda_true = 100

# build network
net = MNISTAutoencoderSlimTik(width_dec=8,
                              memory_depth=1000, opt_method='constant',
                              reduction='mean',
                              sumLambda=0, Lambda=lambda_true / num_batch)

# Z = []
# C = []
ZTZ = []
ZTC = []
for i in range(num_batch):
    inputs = train_data[i * batch_size:(i + 1) * batch_size]
    z = net.feature_extractor(inputs)
    z = net.form_full_conv2d_transpose_matrix(z)
    # Z.append(z)
    # C.append(inputs.reshape(-1))

    ZTZ.append((1 / batch_size) * (z.t() @ z))
    ZTC.append((1 / batch_size) * (z.t() @ inputs.reshape(-1)))

# Z = torch.cat(Z, dim=0)
# C = torch.cat(C)
ZTZ = sum(ZTZ)
ZTC = sum(ZTC)

# solve regularized least squares problem
# u, s, v = torch.svd(Z)
# w_true = v @ (torch.diag(s / (s ** 2 + lambda_true)) @ (u.t() @ C))

u2, s2, _ = torch.svd(ZTZ)
w_true2 = u2 @ torch.diag(1.0 / (s2.to(u2.dtype) + lambda_true)) @ u2.t() @ ZTC

# solve using sTik
err = [torch.norm(w_true2 - torch.cat((net.W.reshape(-1), net.b))).item()]
for i in range(num_batch):
    inputs = train_data[i * batch_size:(i + 1) * batch_size]
    tmp = net(inputs, inputs)
    err.append(torch.norm(w_true2 - torch.cat((net.W.reshape(-1), net.b))).item())


#%%

plt.figure()
plt.semilogy(torch.tensor(err).detach() / torch.norm(w_true2.detach()))
plt.show()




