from slimtik_functions.linear_operators import *
from slimtik_functions import golub_kahan_lanczos_bidiagonalization as gkl
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(20)


def test_hybrid_lsqr(linOp, b, max_iter, x_true, verbose=True):

    x, info = gkl.hybrid_lsqr_gcv(linOp, b, max_iter, x_true=x_true, RegParam='gcv',
                                  tik=True, verbose=True, dtype=torch.float64)

    err = torch.norm(x.reshape(-1) - x_true.reshape(-1)) / torch.norm(x_true)

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
        for k in range(max_iter):
            print('(m, n, k)=(%d, %d, %d)\tres = %0.2e\terr = %0.2e' %
                  (m, n, k + 1, info['Rnrm'][k], info['Enrm'][k]))

    return info

# ==================================================================================================================== #
print('Dense')
m = 1000
n = 500

# A = torch.randn(m, n)

# ill-posed A
u, _ = torch.qr(torch.randn(m, m))
v, _ = torch.qr(torch.randn(n, n))

r = min(m, n)
s = torch.logspace(3, -3, r)
A = u[:, :r] @ torch.diag(s) @ v[:, :r].t()

linOp = DenseMatrix(A)

# ==================================================================================================================== #
# print('Affine')
# m = 20
# n = 10
# bias = True
#
# M = torch.randn(m, n)
# linOp = AffineOperator(M, bias=bias)
#
# if bias:
#     b = torch.eye(m)
#     M = torch.cat((M, b), dim=1)

# ==================================================================================================================== #
# print('Convolution2D')
# N, C_in, H, W = 11, 5, 20, 16
# C_out = 4
# kH, kW = 3, 3
#
# bias = False
# stride = (3, 1)
# padding = 3
# dilation = 2
# groups = 1  # not yet implemented
# alpha = 0.5
#
#
# # initialize
# M = torch.randn(N, C_in, H, W)
#
# linOp = Convolution2D(M, C_in, C_out, (kH, kW),
#                       bias=bias,
#                       stride=stride,
#                       padding=padding,
#                       dilation=dilation,
#                       groups=groups,
#                       alpha=alpha)

# ==================================================================================================================== #
# print('ConvolutionTranspose2D')
# N, C_in, H, W = 7, 5, 8, 8
# C_out = 3
#
# kH = 5
# kW = 3
#
#
# bias = True
# stride = (4, 2)  # assume H - kH is a multiple of s1, likewise forrr W
# output_padding = (1, 1)  # must be smaller than either stride or dilation
# padding = (1, 0)  # must be smaller than kernel size
# dilation = (1, 2)
# groups = 1
# alpha = 3
#
# # initialize
# M = torch.randn(N, C_in, H, W)
#
# linOp = ConvolutionTranspose2D(M, C_in, C_out, (kH, kW),
#                                bias=bias,
#                                stride=stride,
#                                padding=padding,
#                                output_padding=output_padding,
#                                dilation=dilation,
#                                groups=groups,
#                                alpha=alpha)

# ==================================================================================================================== #
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
# linOp = ConcatenatedLinearOperator((linOp2, linOp1, linOp3))

# ==================================================================================================================== #
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
#
# m = 2
# n = linOp1.numel_in()
# M = torch.randn(m, n)
# linOp2 = DenseMatrix(M)
# linOp3 = IdentityMatrix(n, alpha=1)
#
# linOp = ConcatenatedLinearOperator((linOp1, linOp2, linOp3))

# ==================================================================================================================== #
x_true = torch.randn(linOp.numel_in())
b = linOp.A(x_true)
noise = 1e-2 * torch.norm(x_true) * torch.randn_like(b)

info = test_hybrid_lsqr(linOp, b + noise, min(1000, x_true.numel()), x_true, verbose=False)

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
