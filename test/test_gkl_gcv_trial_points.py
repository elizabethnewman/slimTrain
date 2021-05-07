import torch
from slimtik_functions import golub_kahan_lanczos_bidiagonalization as gkl
from slimtik_functions.linear_operators import DenseMatrix

# ==================================================================================================================== #
m = 10
n = 10
k = 5

M = torch.randn(m, n)

linOp = DenseMatrix(M)
b = torch.randn(linOp.numel_out())

U, B, V = gkl.lanczos_bidiagonalization(linOp, b, k)

beta = torch.norm(b)
mb, nb = B.shape
u, s, vh = torch.svd(B, some=False)

e1 = torch.zeros(B.shape[0])
e1[0] = 1.0

alpha = torch.logspace(0, -4, 10)
g_true = torch.zeros(alpha.numel())
for i in range(10):
    B_alpha_pinv = torch.inverse(B.t() @ B + alpha[i] ** 2 * torch.eye(B.shape[1])) @ B.t()
    M = torch.eye(B.shape[0]) - B @ B_alpha_pinv
    num = k * torch.norm(M @ (beta * e1)) ** 2
    den = torch.trace(M) ** 2
    g_true[i] = num / den


g_new = gkl.gcv_trial_points(alpha, u[0, :], s, beta)

print('|g_true - g_new| / |g_true|= %0.2e' % (torch.norm(g_true - g_new) / torch.norm(g_true)).item())