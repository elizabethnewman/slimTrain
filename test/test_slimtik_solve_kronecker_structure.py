import torch
import math
from slimtik_functions.slimtik_solve_kronecker_structure import solve

torch.manual_seed(20)
torch.set_default_dtype(torch.float64)


def vec(x):
    # vectorize column-wise
    return x.t().reshape(-1)

# TODO: add another test with ideal solution
# ==================================================================================================================== #
# expect this test to pass when not using sgcv to find best Lambda (best does not necessarily mean smaller residual)
n_target = 3
n_calTk = 10
n_out = 5
r = 2
sumLambda = 5e-2
Lambda = 1e-2
reduction = 'mean'

beta = 1.0
if reduction == 'mean':
    beta = 1 / math.sqrt(n_calTk)

I = torch.eye(n_target)

# problem setup
M = torch.empty(0)
B = torch.empty(0)
A = torch.empty(0)
for i in range(r + 1):
    Z = torch.randn(n_out, n_calTk)
    C = torch.randn(n_target, n_calTk)
    M = torch.cat((M, Z), dim=1)
    B = torch.cat((B, C), dim=1)
    A = torch.cat((A, torch.kron(Z.t().contiguous(), I)), dim=0)

b = vec(B)
W = torch.randn(n_target, n_out)

# true solution
Iw = torch.eye(W.numel())
wk = vec(W)
bk = vec(B[:, -n_calTk:])
Ak = A[-n_calTk * n_target:, :]

AI = torch.cat((beta * A, math.sqrt(sumLambda) * Iw), dim=0)
z = torch.zeros(A.shape[0] - n_calTk * n_target)
res = Ak @ wk - bk
bI = torch.cat((z, beta * res, Lambda / math.sqrt(sumLambda) * wk))
sk = torch.pinverse(AI) @ bI
w_true = wk - sk
res_true = Ak @ w_true - bk

# without kronecker structure
Zk = M[:, -n_calTk:]
Ck = B[:, -n_calTk:]
Rk = W @ Zk - Ck

# make sure the data is the same
print('Check data:')
print('|Ak - kron(Zk.t(), I)| = %0.4e' % (torch.norm(torch.kron(Zk.t().contiguous(), I) - Ak) / torch.norm(Ak)).item())
print('|vec(Ck) - bk| = %0.4e' % (torch.norm(vec(Ck) - bk) / torch.norm(bk)).item())
print('|vec(Rk) - res| = %0.4e' % (torch.norm(vec(Rk) - res) / torch.norm(res)).item())

W_new, info = solve(Zk, Ck, M, W, sumLambda,
              dtype=torch.float64, opt_method=None, reduction=reduction,
              lower_bound=1e-7, upper_bound=1e-3, Lambda=Lambda)

# compare residuals
print('Check results:')
print('|W_new - w_true| / |w_true| = %0.4e' % (torch.norm(w_true - vec(W_new)) / torch.norm(w_true)).item())
print('|res_true| / |C| = %0.4e' % (torch.norm(res_true) / torch.norm(Ck)).item())
print('|res_best| / |C| = %0.4e' % info['Rnrm'])
print('|res_true - res_best| / |res_true| = %0.4e' %
      torch.abs((torch.norm(res_true) / torch.norm(Ck)) - info['Rnrm']).item())
print('(Lambda, Lambda_best) = (%0.2e, %0.2e)' % (Lambda, info['LambdaBest']))