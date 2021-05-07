import torch
import torch.nn as nn
import math
from slimtik_functions.tikhonov_parameters import sgcv

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
                self.alpha = self.sumLambda / self.iter
                self.alphaHist += [self.alpha]

        # reorient for pytorch
        return (self.W @ x).transpose(0, 1)

    def solve(self, Z, C, dtype=torch.float64):
        orig_dtype = Z.dtype
        with torch.no_grad():
            beta = 1.0
            if self.reduction == 'mean':
                beta = 1 / math.sqrt(Z.shape[1])

            U, S, _ = torch.svd(beta * self.M.to(dtype))

            Z = beta * Z.to(dtype)
            C = beta * C.to(dtype)
            W = self.W.to(dtype)
            WZC = W @ Z - C

            if self.opt_method == 'trial_points':
                ZU = Z.t() @ U
                WZCZU = WZC @ ZU
                WU = W @ U

                n_target, n_calTk = C.shape

                # choose candidates
                n_high = self.upper_bound - self.sumLambda

                if n_high <= 0:
                    # sumLambda is already greater than or equal to upper_bound
                    # don't increase regularization parameter
                    Lambda1 = torch.empty(0)
                else:
                    eps = torch.finfo(Z.dtype).eps
                    Lambda1 = torch.logspace(math.log10(eps), math.log10(n_high), 100)

                n_low = self.sumLambda / 2 - self.lower_bound  # divide by 2 to avoid numerical issues
                if n_low <= 0:
                    # sumLambda is already less than or equal to lower_bound
                    # don't decrease regularization parameter
                    Lambda2 = torch.empty(0)
                else:
                    eps = torch.finfo(Z.dtype).eps
                    Lambda2 = -torch.logspace(math.log10(n_low), math.log10(eps), 100)

                Lambda = torch.cat((Lambda2, Lambda1), dim=0)

                # approximate minimum of gcv function
                f = sgcv(Lambda, ZU, WZC, WZCZU, WU, S, self.sumLambda, n_calTk, n_target)
                idx = torch.argmin(f, dim=0)

                # store
                self.Lambda = Lambda[idx].item()
                self.LambdaHist.append(self.Lambda)

            # update W using Sherman-Morrison-Woodbury
            if self.sumLambda + self.Lambda <= 0:
                raise ValueError('sumLambda must be positive!')

            self.sumLambda += self.Lambda
            alpha = 1.0 / self.sumLambda
            I = torch.eye(W.shape[1])
            s2 = S ** 2
            Binv = alpha * I - alpha ** 2 * ((U @ torch.diag(s2 / (1.0 + alpha * s2))) @ U.t())
            self.W -= ((WZC @ Z.t() + self.Lambda * W) @ Binv).to(orig_dtype)

    def print_outs(self):
        results = {
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': (torch.norm(self.W).item(), self.Lambda, self.sumLambda / (self.iter + 1), self.iter, self.memory_depth)
        }

        return results