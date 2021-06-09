import torch
from slimtik_functions.linear_operators import *
from slimtik_functions.get_convolution_matrix import get_Conv2DTranspose_matrix

# https://arxiv.org/pdf/1603.07285.pdf

torch.manual_seed(20)
torch.set_default_dtype(torch.float32)


def list_squeeze(input):

    for i in range(len(input)):
        input[i] = input[i].squeeze()
    return input


def get_full_matrix(linOp):
    A_mat = torch.empty(0)
    for i in range(linOp.numel_in()):
        ei = torch.zeros(linOp.numel_in())
        ei[i] = 1.0
        Aei = linOp.A(ei)
        A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

    return A_mat


def get_full_matrix_v2(linOp):
    C_in, C_out, kH, kW = linOp.shape_in
    N = linOp.data.shape[0]

    d = linOp.data
    d_shape = d.shape
    linOp.data = d.reshape(-1, 1, d.shape[2], d.shape[3])
    linOp.shape_in = (1, C_out, kH, kW)
    A_mat = torch.zeros(kH, kW, N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])
    A_mat2 = torch.empty(0)
    for i in range(kH):
        for j in range(kW):
            ei = torch.zeros(1, C_out, kH, kW)
            ei[:, :, i, j] = 1.0
            Aei = linOp.A(ei)
            A_mat[i, j] = Aei.reshape(N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])

    linOp.shape_in = (C_in, C_out, kH, kW)
    linOp.data = d.reshape(d_shape)
    A_mat2 = A_mat
    return A_mat2



# ==================================================================================================================== #
print('ConvolutionTranspose2D')
N, C_in, H, W = 11, 16, 7, 7
C_out = 1

kH = 4
kW = 4


bias = True
stride = (2, 2)  # assume H - kH is a multiple of s1, likewise forrr W
output_padding = (0, 0)  # must be smaller than either stride or dilation
padding = (0, 0)  # must be smaller than kernel size
dilation = (1, 1)
groups = 1
alpha = 1

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

A_mat = get_full_matrix(linOp)
# A_mat2 = get_full_matrix_v2(linOp)

# tmp = A_mat2.reshape(4, 2, 7, 5, 1, 31, 18)
# tmp = tmp.permute(2, 5, 6, 0, 1, 3, 4)
# tmp = tmp.reshape(-1, 40)
# tmp = A_mat2.permute(3, 2, 5, 6, 4, 0, 1)
# tmp = tmp.reshape(C_in, -1, kH * kW)
# tmp = list_squeeze(list(torch.tensor_split(tmp, tmp.shape[0])))
# tmp2 = torch.cat(tmp, dim=1)

A_mat3 = get_Conv2DTranspose_matrix(linOp)
A_mat3 = torch.cat((A_mat3, torch.ones(A_mat3.shape[0], 1)), dim=1)
# print(torch.norm(A_mat - tmp2) / torch.norm(A_mat))
print(torch.norm(A_mat - A_mat3) / torch.norm(A_mat))

print('Done!')