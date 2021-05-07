import torch
import torch.nn as nn
import math
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolve


class SlimTikNetwork(nn.Module):

    def __init__(self, feature_extractor, W, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05):
        super(SlimTikNetwork, self).__init__()

        self.feature_extractor = feature_extractor
        self.W = W
        self.bias = bias

        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda

        self.M = torch.empty(0)

        self.Lambda = self.sumLambda
        self.LambdaHist = []

        self.alpha = None
        self.alphaHist = []

        self.iter = 0

    def forward(self, x, c=None):
        x = self.feature_extractor(x)

        # orient in Matlab code version for now
        x = x.transpose(0, 1)
        nex = x.shape[1]
        if self.bias:
            x = torch.cat((x, torch.ones(1, x.shape[1])), dim=0)

        if c is not None:
            with torch.no_grad():
                c = c.transpose(0, 1)
                if self.iter > self.memory_depth:
                    self.M = self.M[:, nex:]

                self.M = torch.cat((self.M, x), dim=1)

                # solve!
                self.solve(x, c)
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

        # reorient for pytorch
        return (self.W @ x).transpose(0, 1)

    def solve(self, Z, C, dtype=torch.float64):
        with torch.no_grad():
            W, info = tiksolve.solve(Z, C, self.M, self.W, self.sumLambda, Lambda=self.Lambda,
                                     dtype=dtype, opt_method=self.opt_method, reduction=self.reduction,
                                     lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def print_outs(self):
        results = {
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': (torch.norm(self.W).item(), self.Lambda, self.alpha, self.iter, self.memory_depth)
        }

        return results