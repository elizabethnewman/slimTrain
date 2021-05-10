import torch
import torch.nn.functional as F
from math import floor

# for convolutions: https://arxiv.org/pdf/1603.07285.pdf

class LinearOperator:

    def __init__(self, data, alpha=1.0):
        super(LinearOperator, self).__init__()
        self.data = data
        self.alpha = alpha

        if data is not None:
            self.dtype = data.dtype
            self.device = data.device
            self.ndim = data.ndim

        if data is None:
            self.shape_operator = (0,)
        else:
            self.shape_operator = tuple(data.shape)

        self.shape_in = None
        self.shape_out = None

    def A(self, x):
        """
        Can take in vectorized input
        """
        raise NotImplementedError

    def AT(self, x):
        """
        Can take in vectorized input
        """
        raise NotImplementedError

    def numel_in(self):
        raise NotImplementedError

    def numel_out(self):
        raise NotImplementedError

    def num_out_features(self):
        raise NotImplementedError

    def to_(self, d):
        if self.data is not None:
            self.data = self.data.to(d)

            if isinstance(d, torch.dtype):
                self.dtype = d
            elif isinstance(d, str):
                self.device = d


class ConcatenatedLinearOperator:
    def __init__(self, linOpList, alpha=1.0):
        """
        Concatenated vertically by default for now
        :param linOpList:
        :type linOpList:
        """
        super(ConcatenatedLinearOperator, self).__init__()
        self.linOpList = linOpList
        self.shape_in = linOpList[0].shape_in
        self.alpha = alpha

        # must avoid identity case
        self.dtype = None
        self.device = None
        for linOp in self.linOpList:
            if linOp.data is not None:
                self.dtype = linOp.dtype
                self.device = linOp.device
                self.ndim = 1 + linOp.ndim
                break

        for i, _ in enumerate(self.linOpList):
            self.linOpList[i].alpha = self.alpha
            self.linOpList[i].dtype = self.dtype
            self.linOpList[i].device = self.device

    def A(self, x):
        # x is a vector that goes into every operator

        b = torch.empty(0)
        for linOp in self.linOpList:
            b = torch.cat((b, linOp.A(x).view(-1)), dim=0)

        return b

    def AT(self, b):
        # b is an output vector
        # we apply AT to each block and sum
        x = torch.zeros(self.numel_in())

        count = 0
        for linOp in self.linOpList:
            n = linOp.numel_out()
            x += linOp.AT(b[count:count + n])
            count += n
        return x

    def numel_in(self):
        return self.linOpList[0].numel_in()

    def numel_out(self):
        n = 0
        for linOp in self.linOpList:
            n += linOp.numel_out()
        return n

    def to_(self, d):
        for i, _ in enumerate(self.linOpList):
            self.linOpList[i].to_(d)

        if isinstance(d, torch.dtype):
            self.dtype = d
        elif isinstance(d, str):
            self.device = d


class IdentityMatrix(LinearOperator):
    def __init__(self, in_features, alpha=1.0):
        super(IdentityMatrix, self).__init__(data=None, alpha=alpha)
        self.shape_in = (in_features,)
        self.shape_out = (in_features,)

    def A(self, x):
        return self.alpha * x

    def AT(self, x):
        return self.alpha * x

    def numel_in(self):
        return self.shape_in[0]

    def numel_out(self):
        return self.shape_out[0]

    def num_out_features(self):
        return self.shape_out[0]


class DenseMatrix(LinearOperator):

    def __init__(self, data, alpha=1.0):
        super(DenseMatrix, self).__init__(data=data, alpha=alpha)
        self.shape_in = (data.shape[1],)
        self.shape_out = (data.shape[0],)

    def A(self, x):
        return self.alpha * self.data @ x

    def AT(self, x):
        return self.alpha * self.data.t() @ x

    def numel_in(self):
        return self.shape_in[0]

    def numel_out(self):
        return self.shape_out[0]

    def num_out_features(self):
        return self.shape_out[0]


class AffineOperator(LinearOperator):

    def __init__(self, data, bias=False, alpha=1.0):
        super(AffineOperator, self).__init__(data=data, alpha=alpha)

        self.shape_in = (data.shape[1],)
        self.shape_out = (data.shape[0],)
        self.bias = bias

    def A(self, x):
        # TODO: add scalar multiply?  Should we multiply bias?
        n = self.shape_in[0]

        y = self.data @ x[:n]
        if self.bias:
            y += x[n:]

        return y

    def AT(self, x):

        y = self.data.t() @ x

        if self.bias:
            y = torch.cat((y, x))

        return y

    def numel_in(self):
        return self.shape_in[0] + self.shape_out[0] * (self.bias is True)

    def numel_out(self):
        return self.shape_out[0]

    def num_out_features(self):
        return self.shape_out[0]


class Convolution2D(LinearOperator):

    def __init__(self, data, in_channels, out_channels, kernel_size,
                 bias=False, stride=1, padding=0, dilation=1, groups=1, alpha=1.0):
        super(Convolution2D, self).__init__(data=data, alpha=alpha)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.shape_in = (out_channels, in_channels, kernel_size[0], kernel_size[1])

        H_out = floor((data.shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        W_out = floor((data.shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

        self.shape_out = (out_channels, H_out, W_out)

    def A(self, x):

        n = prod(self.shape_in)
        x = x.reshape(-1)

        b = None
        if self.bias:
            b = x[n:]
            x = x[:n]

        z = F.conv2d(self.alpha * self.data, x.reshape(self.shape_in), bias=b, stride=self.stride,
                     padding=self.padding, dilation=self.dilation, groups=self.groups)

        return z

    def AT(self, x):
        x = x.reshape(-1, *self.shape_out)

        z = F.conv2d(self.alpha * self.data.permute(1, 0, 2, 3), x.permute(1, 0, 2, 3), bias=None,
                     stride=(self.dilation[0], self.dilation[1]),
                     padding=(self.padding[0], self.padding[1]),
                     dilation=(self.stride[0], self.stride[1]),
                     groups=self.groups).permute(1, 0, 2, 3)

        # truncate
        z = z[:self.shape_in[0], :self.shape_in[1], :self.shape_in[2], :self.shape_in[3]]

        z = z.reshape(-1)
        if self.bias:
            z = torch.cat((z, torch.sum(x, dim=(0, 2, 3)).reshape(-1)))

        return z

    def numel_in(self):
        b = 0
        if self.bias:
            b = self.out_channels

        return prod(self.shape_in) + b

    def numel_out(self):
        return self.data.shape[0] * prod(self.shape_out)

    def num_out_features(self):
        return prod(self.shape_out)


class ConvolutionTranspose2D(LinearOperator):

    def __init__(self, data, in_channels, out_channels, kernel_size,
                 bias=False, stride=1, padding=0, output_padding=0, dilation=1, groups=1, alpha=1.0):
        super(ConvolutionTranspose2D, self).__init__(data=data, alpha=alpha)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        self.shape_in = (in_channels, out_channels, kernel_size[0], kernel_size[1])  # may need to switch channels

        H_out = (data.shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) \
                + output_padding[0] + 1
        W_out = (data.shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) \
                + output_padding[1] + 1

        self.shape_out = (out_channels, H_out, W_out)

    def A(self, x):
        n = prod(self.shape_in)
        x = x.reshape(-1)

        b = None
        if self.bias:
            b = x[n:]
            x = x[:n]

        z = F.conv_transpose2d(self.alpha * self.data, x.reshape(self.shape_in), bias=b,
                               stride=self.stride,
                               padding=self.padding,
                               output_padding=self.output_padding,
                               dilation=self.dilation,
                               groups=self.groups)

        return z

    def AT(self, x):

        x = x.reshape(-1, *self.shape_out)

        z = F.conv2d(x.permute(1, 0, 2, 3), self.alpha * self.data.permute(1, 0, 2, 3), bias=None,
                     stride=self.dilation,
                     padding=self.padding,
                     dilation=self.stride,
                     groups=self.groups)
        z = z.permute(1, 0, 2, 3)

        # truncate
        z = z[:self.shape_in[0], :self.shape_in[1], :self.shape_in[2], :self.shape_in[3]]

        z = z.reshape(-1)
        if self.bias:
            z = torch.cat((z, torch.sum(x, dim=(0, 2, 3)).reshape(-1)))

        return z

    def numel_in(self):
        b = 0
        if self.bias:
            b = self.out_channels

        return prod(self.shape_in) + b

    def numel_out(self):
        return self.data.shape[0] * prod(self.shape_out)

    def num_out_features(self):
        return prod(self.shape_out)


# product of list or tuple entries - available in math package for python3.9
def prod(a):
    n = 1
    for ai in a:
        n *= ai
    return n
