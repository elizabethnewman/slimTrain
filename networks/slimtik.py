import torch
import torch.nn as nn
import math
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolve
import slimtik_functions.slimtik_solve as tiksolvevec
import slimtik_functions.golub_kahan_lanczos_bidiagonalization as gkl
import slimtik_functions.linear_operators as lop


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
            x = torch.cat((x, torch.ones(1, x.shape[1], dtype=x.dtype)), dim=0)

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
            beta = 1.0
            if self.reduction == 'mean':
                beta = 1 / math.sqrt(Z.shape[1])
            W, info = tiksolve.solve(beta * Z, beta * C, beta * self.M, self.W, self.sumLambda, Lambda=self.Lambda,
                                     dtype=dtype, opt_method=self.opt_method,
                                     lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def print_outs(self):
        results = {
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(self.W).item(), self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results


class SlimTikNetworkLinearOperator(nn.Module):

    def __init__(self, feature_extractor, linOp, W, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05):
        super(SlimTikNetworkLinearOperator, self).__init__()

        self.feature_extractor = feature_extractor
        self.linOp = linOp
        self.W = W
        self.bias = bias

        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda

        self.M = []

        self.max_iter = 5  # for lsqr

        self.Lambda = self.sumLambda
        self.LambdaHist = []

        self.alpha = 0
        self.alphaHist = []

        self.iter = 0

    def forward(self, x, c=None):
        x = self.feature_extractor(x)
        self.linOp.data = x

        if c is not None:
            with torch.no_grad():
                if self.iter > self.memory_depth:
                    self.M = self.M[1:]  # remove oldest

                # form new linear operator and add to history
                self.M.append(self.linOp)

                # solve!
                self.solve(x, c)
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist.append(self.alpha)

        return self.linOp.A(self.W)

    def solve(self, Z, c, dtype=torch.float64):
        with torch.no_grad():
            beta = 1.0
            if self.reduction == 'mean':
                beta = 1 / math.sqrt(Z.shape[1])

            # create linear operator
            I = lop.IdentityMatrix(self.W.numel(), alpha=self.sumLambda / beta)
            A = lop.ConcatenatedLinearOperator(self.M + [I])
            A.alpha = beta

            # create right-hand side
            self.linOp.data = Z
            res = self.linOp.A(self.W).reshape(-1, 1).to(self.W.device) - c.reshape(-1, 1).to(self.W.device)
            alpha = self.Lambda / self.sumLambda
            b = torch.cat((res.view(-1), alpha * self.W.view(-1)))
            b = torch.cat((torch.zeros(A.numel_out() - b.numel()), b))

            # only for fixed regularization currently
            s, info = gkl.hybrid_lsqr_gcv(A, b, self.max_iter, tik=True, RegParam=self.Lambda)

            # get new regularization parameters TODO: is this correct?
            self.Lambda = info['RegParamVect'][-1]
            self.LambdaHist.append(self.Lambda)
            self.sumLambda += self.Lambda

            self.W -= s.reshape(self.W.shape)

    def to_(self, device='cpu'):
        self.feature_extractor = self.feature_extractor.to(device)
        self.W = self.W.to(device)
        self.linOp.to_(device)

    def print_outs(self):
        results = {
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(self.W.data).item(), self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results


class SlimTikNetworkLinearOperatorFull(nn.Module):

    def __init__(self, feature_extractor, linOp, W, bias=True,
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05, total_num_batches=1):
        super(SlimTikNetworkLinearOperatorFull, self).__init__()

        self.feature_extractor = feature_extractor
        self.linOp = linOp
        self.W = W
        self.bias = bias

        self.memory_depth = memory_depth
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.opt_method = opt_method
        self.reduction = reduction
        self.sumLambda = sumLambda
        self.total_num_batches = total_num_batches

        self.M = []

        self.Lambda = self.sumLambda
        self.LambdaHist = []

        self.alpha = 0
        self.alphaHist = []

        self.iter = 0

    def forward(self, x, c=None):
        x = self.feature_extractor(x)
        self.linOp.data = x

        if c is not None:
            with torch.no_grad():
                if self.iter > self.memory_depth:
                    self.M = self.M[1:]  # remove oldest

                # form new linear operator and add to history
                self.M.append(self.linOp)

                # solve!
                self.solve(c.reshape(c.shape[0], -1))
                self.iter += 1

                # should we multiply by the total number of batches? self.total_num_batches *
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist.append(self.alpha)

        return self.linOp.A(self.W)

    def solve(self, C, dtype=torch.float32):
        with torch.no_grad():
            n_target = C.shape[1]

            M = lop.ConcatenatedLinearOperator(self.M)
            M = self.get_full_matrix(M)

            n_calTk = self.linOp.data.shape[0]
            num_Z = self.linOp.numel_out()
            Z = M[-num_Z:]
            C = C.reshape(-1, 1)
            beta = 1.0
            if self.reduction == 'mean':
                beta = 1 / math.sqrt(Z.shape[1])

            W, info = tiksolvevec.solve(beta * Z, beta * C, beta * M, self.W, self.sumLambda, n_calTk, n_target,
                                        Lambda=self.Lambda, dtype=dtype, opt_method=self.opt_method,
                                        lower_bound=self.lower_bound, upper_bound=self.upper_bound)

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
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth'),
            'frmt': '{:<15.4e}{:<15.4e}{:<15.4e}{:<15d}{:<15d}',
            'val': [torch.norm(self.W.data).item(), self.Lambda, self.alpha, self.iter, self.memory_depth]
        }

        return results