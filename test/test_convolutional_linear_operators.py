import torch
import torch.nn as nn
from slimtik_functions.linear_operators import Convolution2D, ConvolutionTranspose2D


torch.manual_seed(20)
torch.set_default_dtype(torch.float32)

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
alpha = 1

# initialize data
Z = torch.randn(N, C_in, H, W)

# initialize weights
conv = nn.Conv2d(C_in, C_out, (kH, kW), stride=stride, padding=padding, dilation=dilation, bias=bias)
W = conv.weight.data
b = conv.bias.data
W = torch.cat((W.reshape(-1), b.reshape(-1)))

linOp = Convolution2D(Z, C_in, C_out, (kH, kW),
                      bias=bias,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      alpha=alpha)

# compare
Z1 = conv(Z)
Z2 = linOp.A(W)

err = torch.norm(Z1 - Z2) / torch.norm(Z1)
print('err = %0.2e' % err)


# ==================================================================================================================== #
print('ConvolutionTranspose2D')
N, C_in, H, W = 7, 16, 14, 14
C_out = 1

kH = 4
kW = 4


bias = True
stride = (2, 2)  # assume H - kH is a multiple of s1, likewise forrr W
output_padding = (0, 0)  # must be smaller than either stride or dilation
padding = (1, 1)  # must be smaller than kernel size
dilation = (1, 1)
groups = 1
alpha = 1

# initialize data
Z = torch.randn(N, C_in, H, W)

# initialize weights
convt = nn.ConvTranspose2d(C_in, C_out, (kH, kW), stride=stride, padding=padding, dilation=dilation, bias=bias,
                           output_padding=output_padding)
W = convt.weight.data
b = convt.bias.data
W = torch.cat((W.reshape(-1), b.reshape(-1)))

linOp = ConvolutionTranspose2D(Z, C_in, C_out, (kH, kW),
                               bias=bias,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding,
                               dilation=dilation,
                               groups=groups,
                               alpha=alpha)

# compare
Z1 = convt(Z)
Z2 = linOp.A(W)

err = torch.norm(Z1 - Z2) / torch.norm(Z1)
print('err = %0.2e' % err)
