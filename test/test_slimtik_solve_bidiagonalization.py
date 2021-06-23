from slimtik_functions.linear_operators import *
from old_code.slimtik_functions import slimtik_solve_bidiagonalization as bdiagSolve
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(20)


def test_bidiagonalization_solve(linOpA, linOpM, b, max_iter, x_true, verbose=True):

    x, info = bdiagSolve.solve_bidiagonalization_sgcv(linOpA, b, linOpM, max_iter,
                                                      x_true=x_true, RegParam=1e-2,
                                                      tik=True, verbose=True, dtype=torch.float64)

    err = torch.norm(x.reshape(-1) - x_true.reshape(-1)) / torch.norm(x_true)

    if isinstance(linOpA, torch.Tensor) and linOpA.ndim == 2:
        linOpA = DenseMatrix(linOpA)

    if isinstance(linOpM, torch.Tensor) and linOpM.ndim == 2:
        linOpM = DenseMatrix(linOpM)

    linOp = ConcatenatedLinearOperator((linOpM, linOpA))

    if isinstance(linOp, torch.Tensor) and linOp.ndim == 2:
        res = torch.norm(b.reshape(-1) - (linOp @ x).reshape(-1)) / torch.norm(b)
        m = linOp.shape[0]
        n = linOp.shape[1]
    else:
        res = torch.norm(b.reshape(-1) - linOp.A(x).reshape(-1)) / torch.norm(b)

        m = linOp.numel_out()
        n = linOp.numel_in()

    eps = torch.finfo(x.dtype).eps
    if torch.abs(err - info['Enrm'][-1]) > 2 * eps:
        print('err diff = 0.4%d' % torch.abs(err - info['Enrm'][-1]).item())
    if torch.abs(res - info['Rnrm'][-1]) < 2 * eps:
        print('res diff = 0.4%d' % torch.abs(res - info['Rnrm'][-1]).item())

    if verbose:
        for k in range(1, len(info['Rnrm'])):
            print('(m, n, k)=(%d, %d, %d)\tres = %0.2e\terr = %0.2e' %
                  (m, n, k, info['Rnrm'][k], info['Enrm'][k]))

    return info


# ==================================================================================================================== #
print('Dense')
m = 10
n = 20

A = torch.randn(m, n)
linOpA = DenseMatrix(A)

M = torch.randn(4 * m, n)
linOpM = DenseMatrix(M)

# # ==================================================================================================================== #
# print('Concatenated Linear Operator - Version 1')
# m = 100
# n = 50
# bias = True
# m2 = 6 * (bias is True)
#
# M = torch.randn(m, n)
# linOp1 = DenseMatrix(M, alpha=2)
# linOp2 = IdentityMatrix(n, alpha=0.5)
#
# M2 = torch.randn(m2, n - m2)
# linOp3 = AffineOperator(M2, bias=bias)
#
# if bias:
#     b = torch.eye(m2)
#     M2 = torch.cat((M2, b), dim=1)
#
# linOpM = ConcatenatedLinearOperator((linOp2, linOp1))
# linOpA = linOp3

# # ==================================================================================================================== #
# print('Concatenated Linear Operator - Version 2')
#
# N, C_in, H, W = 7, 5, 20, 16
# C_out = 4
# kH, kW = 3, 3
#
# bias = True
# stride = (3, 1)
# padding = 3
# dilation = 2
# groups = 1  # not yet implemented
#
# # initialize
# M = torch.randn(N, C_in, H, W)
#
# linOp1 = Convolution2D(M, C_in, C_out, (kH, kW),
#                        bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
# linOp2 = Convolution2D(M, C_in, C_out, (kH, kW),
#                        bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#
# m = 2
# n = linOp1.numel_in()
# M = torch.randn(m, n)
# linOp3 = DenseMatrix(M)
# linOp4 = IdentityMatrix(n, alpha=1)
#
# linOpM = ConcatenatedLinearOperator((linOp1, linOp3, linOp4))
# linOpA = linOp2

# ==================================================================================================================== #
linOp = ConcatenatedLinearOperator((linOpM, linOpA))
x_true = torch.randn(linOp.numel_in())
b = linOp.A(x_true)
noise = 0 * torch.norm(x_true) * torch.randn_like(b)

info = test_bidiagonalization_solve(linOpA, linOpM, b + noise, min(1000, x_true.numel()), x_true, verbose=True)

# plot
plt.figure(1)
plt.semilogy(info['Rnrm'])
plt.xlabel('iter')
plt.ylabel('rel. res.')
plt.title('residual')
plt.show()

plt.figure(2)
plt.semilogy(info['Enrm'])
plt.xlabel('iter')
plt.ylabel('rel. err.')
plt.title('error')
plt.show()
