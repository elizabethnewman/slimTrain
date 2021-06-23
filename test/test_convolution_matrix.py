from slimtik_functions.linear_operators import *
from old_code.slimtik_functions.get_convolution_matrix import get_Conv2DTranspose_matrix

# https://arxiv.org/pdf/1603.07285.pdf

torch.manual_seed(20)
torch.set_default_dtype(torch.float32)


def get_full_matrix(linOp):
    A_mat = torch.empty(0)
    for i in range(linOp.numel_in()):
        ei = torch.zeros(linOp.numel_in())
        ei[i] = 1.0
        Aei = linOp.A(ei)
        A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

    return A_mat


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

A_mat3 = get_Conv2DTranspose_matrix(linOp)
print(torch.norm(A_mat - A_mat3) / torch.norm(A_mat))

print('Done!')