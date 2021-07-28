import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import slimtik_functions.slimtik_solve as tiksolvevec


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        # shape is a tuple
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MNISTAutoencoderFeatureExtractor(nn.Module):

    def __init__(self, width_enc=16, width_dec=16, intrinsic_dim=50):
        super(MNISTAutoencoderFeatureExtractor,self).__init__()

        enc1 = nn.Conv2d(1, width_enc, (4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        enc2 = nn.Conv2d(width_enc, 2 * width_enc, (4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        act2 = nn.ReLU()
        encr = View((-1, 7 * 7 * 2 * width_enc,))
        enc3 = nn.Linear(7 * 7 * 2 * width_enc, intrinsic_dim, bias=True)
        self.enc = nn.Sequential(enc1, act1, enc2, act2, encr, enc3)

        dec0 = nn.Linear(intrinsic_dim, 7 * 7 * 2 * width_dec, bias=True)
        decr = View((-1, width_dec * 2, 7, 7))
        decb1 = nn.BatchNorm2d(width_dec * 2)
        dec1 = nn.ConvTranspose2d(2 * width_dec, width_dec, (4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        decb2 = nn.BatchNorm2d(width_dec)
        self.dec_feature_extractor = nn.Sequential(dec0, decr, decb1, dec1, act1, decb2)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec_feature_extractor(x)
        return x


class MNISTAutoencoder(nn.Module):

    def __init__(self, width_enc=16, width_dec=16, intrinsic_dim=50):
        super(MNISTAutoencoder, self).__init__()

        self.feature_extractor = MNISTAutoencoderFeatureExtractor(width_enc, width_dec, intrinsic_dim)
        self.final_layer = nn.ConvTranspose2d(width_dec, 1, (4, 4), stride=(2, 2), padding=(1, 1), bias=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.final_layer(x)
        return x


class MNISTAutoencoderSlimTik(nn.Module):
    def __init__(self, width_enc=16, width_dec=16, intrinsic_dim=50, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, device='cpu'):
        super(MNISTAutoencoderSlimTik, self).__init__()

        # Pytorch network
        self.feature_extractor = MNISTAutoencoderFeatureExtractor(width_enc, width_dec, intrinsic_dim)
        self.device = device

        # final convolution transpose layer and parameters
        self.final_layer = dict()
        self.final_layer['in_channels'] = width_dec
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

        self.W = W.to(self.device)
        self.W_shape = W.shape
        self.b = b.to(self.device)

        self.Wb_diff = torch.zeros(W.numel() + b.numel())

        # slimtik parameters
        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda

        # store history
        self.M = torch.empty(0, device=device)

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

        if self.training:
            with torch.no_grad():
                z = self.form_full_conv2d_transpose_matrix(x).to(self.device)

                # get batch size
                n_calTk = x.shape[0]

                # store history
                if self.iter > self.memory_depth:
                    self.M = self.M[z.shape[0]:]  # remove oldest

                # form new linear operator and add to history
                self.M = torch.cat((self.M, z), dim=0)

                # solve for W and b
                self.solve(z, c.to(self.device), n_calTk)

                # update regularization parameter and iteration
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

        return F.conv_transpose2d(x, self.W.to(x.device), bias=self.b.to(x.device),
                                  stride=self.final_layer['stride'], padding=self.final_layer['padding'])

    def solve(self, Z, C, n_calTk, dtype=torch.float32):
        n_target = C.shape[1]
        C = C.reshape(-1, 1)

        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(Z.shape[1])

        W_old = torch.cat((self.W.reshape(-1), self.b.reshape(-1)))
        W, info = tiksolvevec.solve(beta * Z, beta * C, (beta * self.M).t() @ (beta * self.M), deepcopy(W_old),
                                    self.sumLambda, n_calTk, n_target,
                                    Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method, device=self.device,
                                    lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W[:-1].reshape(self.W_shape)
        self.b = W[-1]
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def form_full_conv2d_transpose_matrix(self, x):
        C_in = self.final_layer['in_channels']
        C_out = self.final_layer['out_channels']
        kH, kW = self.final_layer['kernel_size']
        shape_out = self.final_layer['shape_out']
        stride = self.final_layer['stride']
        padding = self.final_layer['padding']

        # number of samples
        N = x.shape[0]
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])

        A_mat = torch.zeros(kH, kW, N, C_in, C_out, shape_out[0], shape_out[1], device=x.device)
        for i in range(kH):
            for j in range(kW):
                ei = torch.zeros(1, C_out, kH, kW, device=x.device)
                ei[:, :, i, j] = 1.0
                Aei = F.conv_transpose2d(x, ei, bias=None, stride=stride, padding=padding)
                A_mat[i, j] = Aei.reshape(N, C_in, C_out, shape_out[0], shape_out[1])

        # reshape
        A_mat = A_mat.permute(3, 2, 5, 6, 4, 0, 1)
        A_mat = A_mat.reshape(C_in, -1, kH * kW)
        A_mat = self.list_squeeze(list(torch.tensor_split(A_mat, A_mat.shape[0])))
        A_mat = torch.cat(A_mat, dim=1)

        # add bias
        A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0], 1, device=x.device)), dim=1)

        return A_mat

    def to_(self, device='cpu'):
        self.feature_extractor = self.feature_extractor.to(device)
        self.W = self.W.to(device)
        self.b = self.b.to(device)

    def clear_(self):
        self.M = None
        self.Wb_diff = None

    @staticmethod
    def list_squeeze(input):

        for i in range(len(input)):
            input[i] = input[i].squeeze()
        return input

    def print_outs(self):
        results = {
            'str': ('|W|', '|W-W_old|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(torch.cat((self.W.data.reshape(-1), self.b.data))).item(),
                    torch.norm(self.Wb_diff).item(),
                    self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results