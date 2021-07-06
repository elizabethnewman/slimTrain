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


class MNISTAutoencoder(nn.Module):

    def __init__(self, width=16, intrinsic_dim=50):
        super(MNISTAutoencoder,self).__init__()

        enc1 = nn.Conv2d(in_channels=1, out_channels=width,
                         kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        enc2 = nn.Conv2d(in_channels=width, out_channels=2 * width,
                         kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        act2 = nn.ReLU()
        encr = View((-1, 7 * 7 * 2 * width,))
        enc3 = nn.Linear(in_features=7 * 7 * 2 * width, out_features=intrinsic_dim, bias=True)

        self.enc = nn.Sequential(enc1, act1, enc2, act2, encr, enc3)

        dec0 = nn.Linear(in_features=intrinsic_dim, out_features=7 * 7 * 2 * width, bias=True)
        decr = View((-1, width * 2, 7, 7))
        decb1 = nn.BatchNorm2d(width * 2)
        dec1 = nn.ConvTranspose2d(in_channels=2 * width, out_channels=width,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        decb2 = nn.BatchNorm2d(width)
        dec2 = nn.ConvTranspose2d(in_channels=width, out_channels=1,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        self.dec = nn.Sequential(dec0, decr, decb1, dec1, act1, decb2, dec2)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class MNISTAutoencoderFeatureExtractor(nn.Module):

    def __init__(self, width=16, intrinsic_dim=50):
        super(MNISTAutoencoderFeatureExtractor,self).__init__()

        enc1 = nn.Conv2d(in_channels=1, out_channels=width,
                         kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        enc2 = nn.Conv2d(in_channels=width, out_channels=2 * width,
                         kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        act2 = nn.ReLU()
        encr = View((-1, 7 * 7 * 2 * width,))
        enc3 = nn.Linear(in_features=7 * 7 * 2 * width, out_features=intrinsic_dim, bias=True)

        self.enc = nn.Sequential(enc1, act1, enc2, act2, encr, enc3)

        dec0 = nn.Linear(in_features=intrinsic_dim, out_features=7 * 7 * 2 * width, bias=True)
        decr = View((-1, width * 2, 7, 7))
        decb1 = nn.BatchNorm2d(width * 2)
        dec1 = nn.ConvTranspose2d(in_channels=2 * width, out_channels=width,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        decb2 = nn.BatchNorm2d(width)

        self.dec = nn.Sequential(dec0, decr, decb1, dec1, act1, decb2)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class MNISTAutoencoderSlimTik(nn.Module):
    # TODO: add device mapping
    def __init__(self, width=16, intrinsic_dim=50, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, device='gpu'):
        super(MNISTAutoencoderSlimTik, self).__init__()

        # Pytorch network
        self.feature_extractor = MNISTAutoencoderFeatureExtractor(width=width, intrinsic_dim=intrinsic_dim)
        self.device = device

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

        self.W = W.to(self.device)
        self.W_shape = W.shape
        self.b = b.to(self.device)

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
        # z = self.form_full_conv2d_transpose_matrix(x).to(self.device)
        # z3 = self.form_full_matrix(x).to(self.device)

        if self.training:
            with torch.no_grad():
                z = self.form_full_conv2d_transpose_matrix2(x).to(self.device)
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
                self.solve(z, c.to(self.device), n_calTk)

                # update regularization parameter and iteration
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

        # # for printing only
        # self.Wb_grad += z.t() @ (z @ torch.cat((self.W.reshape(-1), self.b)) - c.reshape(-1).to(self.device)) +\
        #               self.Lambda * torch.cat((self.W.reshape(-1), self.b))

        return F.conv_transpose2d(x, self.W.to(x.device), bias=self.b.to(x.device),
                                  stride=self.final_layer['stride'], padding=self.final_layer['padding'])

    def solve(self, Z, C, n_calTk, dtype=torch.float32):
        n_target = C.shape[1]
        C = C.reshape(-1, 1)

        beta = 1.0
        if self.reduction == 'mean':
            beta = 1 / math.sqrt(Z.shape[1])

        W_old = torch.cat((self.W.reshape(-1), self.b.reshape(-1)))
        W, info = tiksolvevec.solve(beta * Z, beta * C, beta * self.M, deepcopy(W_old), self.sumLambda, n_calTk, n_target,
                                    Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method, device=self.device,
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

    def form_full_conv2d_transpose_matrix2(self, x):
        C_in = self.final_layer['in_channels']
        C_out = self.final_layer['out_channels']
        kH, kW = self.final_layer['kernel_size']
        shape_out = self.final_layer['shape_out']
        stride = self.final_layer['stride']
        padding = self.final_layer['padding']

        # number of samples
        N = x.shape[0]
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])

        ei = torch.eye(kH * kW, device=x.device).reshape(1, kH * kW, kH, kW)
        A_mat = F.conv_transpose2d(x, ei, bias=None, stride=stride, padding=padding)

        A_mat = A_mat.reshape(N, -1, ei.shape[1], shape_out[0], shape_out[1])
        A_mat = A_mat.permute(0, 3, 4, 1, 2)
        A_mat = A_mat.reshape(-1, (kH * kW) ** 2)
        A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0], 1, device=x.device)), dim=1)
        return A_mat

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

    def form_full_matrix(self, x):
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

    @staticmethod
    def list_squeeze(input):

        for i in range(len(input)):
            input[i] = input[i].squeeze()
        return input

    def print_outs(self):
        results = {
            'str': ('|W|', '|W-W_old|', '|grad_W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(torch.cat((self.W.data.reshape(-1), self.b.data))).item(),
                    torch.norm(self.Wb_diff).item(), torch.norm(self.Wb_grad).item(),
                    self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results