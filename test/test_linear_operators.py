import torch
from slimtik_functions.linear_operators import *

# https://arxiv.org/pdf/1603.07285.pdf

torch.manual_seed(20)
torch.set_default_dtype(torch.float32)


def test_operator(linOp):
    # matricized version
    A_mat = torch.empty(0)
    for i in range(linOp.numel_in()):
        ei = torch.zeros(linOp.numel_in())
        ei[i] = 1.0
        Aei = linOp.A(ei)
        A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

    err = torch.zeros(10, 2)
    for i in range(10):
        x1 = torch.randn(linOp.numel_in())
        x2 = torch.randn(linOp.numel_out())

        y1 = A_mat @ x1
        y2 = A_mat.t() @ x2

        z1 = linOp.A(x1)
        z2 = linOp.AT(x2)

        err[i, 0] = torch.norm(y1.reshape(-1) - z1.reshape(-1)) / torch.norm(y1)
        err[i, 1] = torch.norm(y2.reshape(-1) - z2.reshape(-1)) / torch.norm(y1)

    err_mean = torch.mean(err, dim=0)
    print('err_A = %0.2e\terr_AT = %0.2e' % (err_mean[0].item(), err_mean[1].item()))

    if isinstance(linOp, DenseMatrix) or isinstance(linOp, AffineOperator):
        print('err_M = %0.2e' % (torch.norm(linOp.alpha * M - A_mat) / torch.norm(M)).item())


# ==================================================================================================================== #
print('Scaled Identity')

n = 3

linOp = IdentityMatrix(n, alpha=2)

test_operator(linOp)

# ==================================================================================================================== #
print('Dense')
m = 5
n = 3
alpha = 2

M = torch.randn(m, n)
linOp = DenseMatrix(M, alpha)

test_operator(linOp)

# ==================================================================================================================== #
print('Affine')
m = 5
n = 3
bias = True

M = torch.randn(m, n)
linOp = AffineOperator(M, bias=bias)

if bias:
    b = torch.eye(m)
    M = torch.cat((M, b), dim=1)

test_operator(linOp)

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


# initialize
M = torch.randn(N, C_in, H, W)

linOp = Convolution2D(M, C_in, C_out, (kH, kW),
                      bias=bias,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      alpha=alpha)

test_operator(linOp)

# ==================================================================================================================== #
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

test_operator(linOp)


# ==================================================================================================================== #
print('Concatenated Linear Operator - Version 1')
m = 2
n = 20
bias = True

M = torch.randn(m, n)
linOp1 = DenseMatrix(M)
linOp2 = IdentityMatrix(n, alpha=0.5)

m2 = 6
M2 = torch.randn(m2, n - m2)
linOp3 = AffineOperator(M2, bias=True)

if bias:
    b = torch.eye(m2)
    M2 = torch.cat((M2, b), dim=1)

linOp = ConcatenatedLinearOperator((linOp2, linOp1, linOp3))

test_operator(linOp)


# ==================================================================================================================== #
print('Concatenated Linear Operator - Version 2')

N, C_in, H, W = 7, 5, 20, 16
C_out = 4
kH, kW = 3, 3

bias = True
stride = (3, 1)
padding = 3
dilation = 2
groups = 1  # not yet implemented
alpha = 2

# initialize
M = torch.randn(N, C_in, H, W)

linOp1 = Convolution2D(M, C_in, C_out, (kH, kW),
                       bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


m = 2
n = linOp1.numel_in()
M = torch.randn(m, n)
linOp2 = DenseMatrix(M)
linOp3 = IdentityMatrix(n, alpha=1)

linOp = ConcatenatedLinearOperator((linOp1, linOp2, linOp3), alpha=2)

test_operator(linOp)
