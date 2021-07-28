import torch
from slimtik_functions import slimtik_solve as tiksolvevec
import matplotlib.pyplot as plt
import math
from copy import deepcopy

torch.manual_seed(20)
torch.set_default_dtype(torch.float64)

n_ex = 600
batch = 10
num_batch = n_ex // batch

n_target = 4
n_out = 3
sumLambda = 0
Lambda_true = 0


# problem setup
A = torch.randn(n_ex * n_target, n_out * n_target)
w0 = torch.randn(n_target * n_out)
c = A @ w0 + 0 * torch.randn(A.shape[0])


# true solution
rhs = torch.cat((c.reshape(-1), torch.zeros(A.shape[1])))
aI = math.sqrt(Lambda_true) * torch.eye(A.shape[1])
w_true, *_ = torch.lstsq(rhs.reshape(-1, 1), torch.cat((A, aI)))
w_true = w_true[:A.shape[1]]

# u, s, v = torch.svd(A)
# w_true = v @ torch.diag(s / (s ** 2 + Lambda_true)) @ u.t() @ c

# main iteration
# M = torch.empty(0)
MtM = torch.zeros(1)

w = torch.randn_like(w0)
w_hist = [torch.clone(w)]
err = torch.zeros(num_batch)
B = torch.zeros(1)
I = torch.eye(A.shape[1])
Lambda = Lambda_true / num_batch
for i in range(num_batch):

    # form batch
    Ab = A[i * batch:(i + 1) * batch]
    cb = c[i * batch:(i + 1) * batch]

    # sTik update
    # B = B + Lambda * I + Ab.t() @ Ab
    # rhs = Ab.t() @ (Ab @ w - cb).reshape(-1) + Lambda * w.reshape(-1)
    # s, *_ = torch.lstsq(rhs.reshape(-1, 1), B)
    # w = w - s.reshape(w.shape)

    # # can only use for slimtik
    MtM = MtM + Ab.t() @ Ab

    w, info = tiksolvevec.solve(Ab, cb, MtM, w, sumLambda, batch, n_target,
                                dtype=torch.float64,
                                opt_method=None, lower_bound=1e-7, upper_bound=1e-3, Lambda=Lambda)
    w = w.reshape(-1)
    Lambda = info['LambdaBest']
    sumLambda = info['sumLambda']

    w_hist.append(torch.clone(w.reshape(-1)))
    err[i] = torch.norm(w.reshape(-1) - w_true.reshape(-1))

#%%

plt.figure()
plt.semilogy(err / torch.norm(w_true))
plt.show()
