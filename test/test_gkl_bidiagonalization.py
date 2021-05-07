import torch
from slimtik_functions.linear_operators import *
from slimtik_functions.golub_kahan_lanczos_bidiagonalization import lanczos_bidiagonalization

torch.set_default_dtype(torch.float64)


def test_bidiag(linOp, b, k):
    U, B, V = lanczos_bidiagonalization(linOp, b, k)

    n = linOp.numel_in()
    A_mat = torch.empty(0)
    for i in range(n):
        ei = torch.zeros(n)
        ei[i] = 1.0
        Aei = linOp.A(ei)
        A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

    # check U and V
    err_u = torch.norm(U.t() @ U - torch.eye(k + 1))
    err_v = torch.norm(V.t() @ V - torch.eye(k))
    err_b = torch.norm(U.t() @ (A_mat @ V) - B)

    print('err_u = %0.2e\terr_v = %0.2e\terr_b = %0.2e' % (err_u.item(), err_v.item(), err_b.item()))

# # ==================================================================================================================== #
# print('Identity')  # TODO: catch this case
# n = 20
# k = 5
#
# linOp = IdentityMatrix(n)
# b = torch.randn(linOp.numel_out())
#
# test_bidiag(linOp, b, k)

# ==================================================================================================================== #
print('Dense')
m = 10
n = 20
k = 5

M = torch.randn(m, n)
linOp = DenseMatrix(M)
b = torch.randn(linOp.numel_out())

test_bidiag(linOp, b, k)

# ==================================================================================================================== #
print('Affine')
m = 10
n = 20
k = 5

M = torch.randn(m, n)
linOp = AffineOperator(M)
b = torch.randn(linOp.numel_out())

test_bidiag(linOp, b, k)


# ==================================================================================================================== #
print('Convolution2D')
N, C_in, H, W = 11, 5, 20, 16
C_out = 4
kH, kW = 3, 3

bias = True
stride = (3, 1)
padding = 3
dilation = 2
groups = 1  # not yet implemented
alpha = 0.5
k = 5


# initialize
M = torch.randn(N, C_in, H, W)

linOp = Convolution2D(M, C_in, C_out, (kH, kW),
                      bias=bias,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      alpha=alpha)

b = torch.randn(linOp.numel_out())

test_bidiag(linOp, b, k)

# # ==================================================================================================================== #
print('ConvolutionTranspose2D')
N, C_in, H, W = 7, 5, 8, 8
C_out = 3

kH = 5
kW = 3


bias = True
stride = (4, 2)  # assume H - kH is a multiple of s1, likewise forrr W
output_padding = (1, 1)  # must be smaller than either stride or dilation
padding = (1, 0)  # must be smaller than kernel size
dilation = (1, 2)
groups = 1
alpha = 3
k = 3

# initialize
M = torch.randn(N, C_in, H, W)

linOp = ConvolutionTranspose2D(M, C_in, C_out, (kH, kW),
                               bias=bias,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding,
                               dilation=dilation,
                               groups=groups,
                               alpha=alpha)

b = torch.randn(linOp.numel_out())
test_bidiag(linOp, b, k)
