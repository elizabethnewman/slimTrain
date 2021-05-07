import torch
from slimtik_functions import golub_kahan_lanczos_bidiagonalization as gkl
from slimtik_functions.linear_operators import DenseMatrix, Convolution2D


def test_hybrid_lsqr(linOp, b, max_iter, x_true):
    for k in range(1, max_iter + 1):
        x, info = gkl.hybrid_lsqr_gcv(linOp, b, k, x_true=x_true, RegParam='gcv', tik=True)

        res = torch.norm(b.reshape(-1) - linOp.A(x).reshape(-1)) / torch.norm(b)
        err = torch.norm(x.reshape(-1) - x_true.reshape(-1)) / torch.norm(x_true)

        print('(m, n, k)=(%d, %d, %d)\tres = %0.2e\terr = %0.2e' %
              (linOp.numel_out(), linOp.numel_in(), k, res.item(), err.item()))


# ==================================================================================================================== #
# print('Dense')
# m = 200
# n = 100
#
# A = torch.randn(m, n)
# x_true = torch.randn(n)
#
# linOp = DenseMatrix(A)
#
# b = linOp.A(x_true)
#
# test_hybrid_lsqr(linOp, b, 30)

# ==================================================================================================================== #
print('Convolution2D')
N, C_in, H, W = 11, 5, 20, 16
C_out = 4
kH, kW = 3, 3

bias = False
stride = (3, 1)
padding = 3
dilation = 2
groups = 1  # not yet implemented
alpha = 0.5


# initialize
M = torch.randn(N, C_in, H, W)

linOp = Convolution2D(M, C_in, C_out, (kH, kW),
                      bias=bias,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      alpha=alpha)

x_true = torch.randn(linOp.shape_in)
b = linOp.A(x_true)

test_hybrid_lsqr(linOp, b, 20, x_true)
