
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from autoencoder.mnist import MNISTAutoencoderFeatureExtractor
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolve
import slimtik_functions.slimtik_solve as tiksolvevec
from slimtik_functions.get_convolution_matrix import get_Conv2DTranspose_matrix
from copy import deepcopy


class SlimTikNetworkMNIST(nn.Module):
    # TODO: add device mapping
    def __init__(self, width=16, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05):
        super(SlimTikNetworkMNIST, self).__init__()

        # Pytorch network
        self.feature_extractor = MNISTAutoencoderFeatureExtractor(width=width)

        # final convolution transpose layer and parameters
        self.final_layer = dict()
        self.final_layer['in_channels'] = 16
        self.final_layer['out_channels'] = 1
        self.final_layer['kernel_size'] = (4, 4)
        self.final_layer['stride'] = (2, 2)
        self.final_layer['padding'] = (1, 1)
        self.final_layer['bias'] = bias
        self.final_layer['shape_out'] = (28, 28)

        # initialize separable weights
        final_layer = nn.ConvTranspose2d(in_channels=self.final_layer['in_channels'],
                                         out_channels=self.final_layer['out_channels'],
                                         kernel_size=self.final_layer['kernel_size'],
                                         stride=self.final_layer['stride'],
                                         padding=self.final_layer['padding'],
                                         bias=self.final_layer['bias'])

        W = final_layer.weight.data
        b = None
        if self.final_layer['bias']:
            b = final_layer.bias.data.reshape(-1)

        self.W = W
        self.W_shape = W.shape
        self.b = b

        self.Wb_diff = torch.zeros(W.numel() + b.numel())
        self.Wb_grad = torch.zeros(W.numel() + b.numel())

        # slimtik parameters
        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda

        # store history
        self.M = torch.empty(0)

        # regularization parameters
        self.Lambda = self.sumLambda
        self.LambdaHist = []

        self.alpha = None
        self.alphaHist = []

        # iteration counter
        self.iter = 0

    def forward(self, x, c=None):

        # extract features
        x = self.feature_extractor(x)

        # form full matrix
        z = form_full_conv2d_transpose_matrix(x)

        if self.training:
            with torch.no_grad():
                # reset gradient of Wb
                self.Wb_grad = torch.zeros(self.W.numel() + self.b.numel())

                # get batch size
                n_calTk = x.shape[0]

                # store history
                if self.iter > self.memory_depth:
                    self.M = self.M[z.shape[0]:]  # remove oldest

                # form new linear operator and add to history
                self.M = torch.cat((self.M, z), dim=0)

                # solve for W and b
                self.solve(z, c, n_calTk)

                # update regularization parameter and iteration
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

        self.Wb_grad += z.t() @ (z @ torch.cat((self.W.reshape(-1), self.b)) - c.reshape(-1)) +\
                      self.Lambda * torch.cat((self.W.reshape(-1), self.b))

        return F.conv_transpose2d(x, self.W, bias=self.b,
                                  stride=self.final_layer['stride'], padding=self.final_layer['padding'])

    def solve(self, Z, C, n_calTk, dtype=torch.float32):
        n_target = C.shape[1]
        C = C.reshape(-1, 1)

        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(Z.shape[1])

        W_old = torch.cat((self.W.reshape(-1), self.b.reshape(-1)))
        W, info = tiksolvevec.solve(beta * Z, beta * C, beta * self.M, deepcopy(W_old), self.sumLambda, n_calTk, n_target,
                                    Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method,
                                    lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W_diff = W.reshape(-1) - W_old.reshape(-1)
        self.W = W[:-1].reshape(self.W_shape)
        self.b = W[-1]
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def to_(self, device='cpu'):
        self.feature_extractor = self.feature_extractor.to(device)
        self.W = self.W.to(device)

    def get_full_matrix(self, x):
        n = math.prod(self.W_shape)
        A_mat = torch.empty(0)
        for i in range(n):
            ei = torch.zeros(n)
            ei[i] = 1.0
            ei = ei.reshape(self.W_shape)
            Aei = F.conv_transpose2d(x, ei, bias=None,
                                     stride=self.final_layer['stride'], padding=self.final_layer['padding'])
            A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

        # add bias
        A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0], 1)), dim=1)

        return A_mat

    def print_outs(self):
        results = {
            'str': ('|W|', '|W-W_old|', '|grad_W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(torch.cat((self.W.data.reshape(-1), self.b.data))).item(),
                    torch.norm(self.Wb_diff).item(), torch.norm(self.Wb_grad).item(),
                    self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results


def form_full_conv2d_transpose_matrix(x, C_in=16, C_out=1, kernel_size=(4, 4),
                                      stride=(2, 2), padding=(1, 1), shape_out=(28, 28)):

    kH, kW = kernel_size

    # number of samples
    N = x.shape[0]
    x = x.reshape(-1, 1, x.shape[2], x.shape[3])

    A_mat = torch.zeros(kH, kW, N, C_in, C_out, shape_out[0], shape_out[1])
    for i in range(kH):
        for j in range(kW):
            ei = torch.zeros(1, C_out, kH, kW)
            ei[:, :, i, j] = 1.0
            Aei = F.conv_transpose2d(x, ei, bias=None, stride=stride, padding=padding)
            A_mat[i, j] = Aei.reshape(N, C_in, C_out, shape_out[0], shape_out[1])

    # reshape
    A_mat = A_mat.permute(3, 2, 5, 6, 4, 0, 1)
    A_mat = A_mat.reshape(C_in, -1, kH * kW)
    A_mat = list_squeeze(list(torch.tensor_split(A_mat, A_mat.shape[0])))
    A_mat = torch.cat(A_mat, dim=1)

    # add bias
    A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0], 1)), dim=1)

    return A_mat

def list_squeeze(input):

    for i in range(len(input)):
        input[i] = input[i].squeeze()
    return input


class SlimTikNetworkLinearOperatorFull(nn.Module):

    def __init__(self, feature_extractor, linOp, W, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, total_num_batches=1):
        super(SlimTikNetworkLinearOperatorFull, self).__init__()

        self.feature_extractor = feature_extractor
        self.linOp = linOp
        self.W = W
        self.W_diff = torch.zeros(1)
        self.W_grad = torch.zeros(1)
        self.bias = bias

        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda
        self.total_num_batches = total_num_batches

        self.M = torch.empty(0)

        self.Lambda = self.sumLambda
        self.LambdaHist = []

        self.alpha = 0
        self.alphaHist = []

        self.iter = 0

    def forward(self, x, c=None):
        x = self.feature_extractor(x)
        self.linOp.data = x
        num_Z = self.linOp.numel_out()
        if c is not None:
            with torch.no_grad():
                if self.iter > self.memory_depth:
                    self.M = self.M[num_Z:]  # remove oldest

                # form new linear operator and add to history
                # self.M.append(self.linOp)
                self.M = torch.cat((self.M, get_Conv2DTranspose_matrix(self.linOp)), dim=0)

                # solve!
                self.solve(c.reshape(c.shape[0], -1))
                self.iter += 1

                # should we multiply by the total number of batches? self.total_num_batches *
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist.append(self.alpha)

        return self.linOp.A(self.W)

    def solve(self, C, dtype=torch.float32):
        n_target = C.shape[1]

        n_calTk = self.linOp.data.shape[0]
        num_Z = self.linOp.numel_out()
        Z = self.M[-num_Z:]
        C = C.reshape(-1, 1)
        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(Z.shape[1])

        W, info = tiksolvevec.solve(beta * Z, beta * C, beta * self.M, self.W.clone(), self.sumLambda, n_calTk, n_target,
                                    Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method,
                                    lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W_diff = W.reshape(-1) - self.W.reshape(-1)
        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def to_(self, device='cpu'):
        self.feature_extractor = self.feature_extractor.to(device)
        self.W = self.W.to(device)
        self.linOp.to_(device)

    @staticmethod
    def get_full_matrix(linOp):
        A_mat = torch.empty(0)
        for i in range(linOp.numel_in()):
            ei = torch.zeros(linOp.numel_in())
            ei[i] = 1.0
            Aei = linOp.A(ei)
            A_mat = torch.cat((A_mat, Aei.view(-1, 1)), dim=1)

        return A_mat

    def print_outs(self):
        results = {
            'str': ('|W|', '|W-W_old|', '|grad_W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(self.W.data).item(), torch.norm(self.W_diff).item(), torch.norm(self.W_grad).item(),
                    self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results