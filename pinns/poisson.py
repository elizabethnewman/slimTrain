import torch
from utils import parameters_norm, grad_norm
import math
import slimtik_functions.slimtik_solve_kronecker_structure as tiksolve


class PoissonPINN:

    def __init__(self, net, device='cpu'):
        super(PoissonPINN, self).__init__()

        self.feature_extractor = net
        self.device = device
        self.iter = 0

        self.loss_u = torch.zeros(1)
        self.loss_b = torch.zeros(1)

    def extract_data(self, data, requires_grad=True):
        x = data['interior'][:, 0:1].clone().detach().requires_grad_(requires_grad).to(self.device)
        y = data['interior'][:, 1:2].clone().detach().requires_grad_(requires_grad).to(self.device)

        xb = data['boundary'][:, 0:1].clone().detach().requires_grad_(requires_grad).to(self.device)
        yb = data['boundary'][:, 1:2].clone().detach().requires_grad_(requires_grad).to(self.device)

        f = data['forcing'].clone().detach().requires_grad_(False).to(self.device)
        g = data['boundary_true'].clone().detach().requires_grad_(False).to(self.device)

        return x, y, xb, yb, f, g

    def net_u(self, x):
        return self.feature_extractor(x)

    def loss(self, x, y, xb, yb, f, g):
        self.iter += 1

        f_pred = self.net_f(x, y)
        ub_pred = self.net_u(torch.cat((xb, yb), dim=1))

        self.loss_u = (0.5 / x.numel()) * torch.norm(f_pred.view(-1) - f.view(-1)) ** 2
        self.loss_b = (0.5 / xb.numel()) * (torch.norm(ub_pred.view(-1) - g.view(-1)) ** 2)

        return self.loss_u + self.loss_b

    def net_f(self, x, y):
        # form Laplacian
        u = self.net_u(torch.cat((x, y), dim=1))

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        lap_u = u_xx + u_yy

        return lap_u

    def print_outs(self):
        results = {
            'str': ('loss_u', 'loss_b'),
            'frmt': '{:<18.4e}{:<18.4e}',
            'val': None
        }

        results['val'] = [self.loss_u.item(), self.loss_b.item()]
        return results


class PoissonPINNSlimTik:

    def __init__(self, feature_extractor, W, bias=True, device='cpu',
                 memory_depth=0, lower_bound=1e-7, upper_bound=1e3,
                 opt_method='trial_points', reduction='mean', sumLambda=0.05):
        super(PoissonPINNSlimTik, self).__init__()

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

        self.device = device
        self.iter = 0

        self.loss_u = torch.zeros(1)
        self.loss_b = torch.zeros(1)

    def extract_data(self, data, requires_grad=True):
        x = data['interior'][:, 0:1].clone().detach().requires_grad_(requires_grad).to(self.device)
        y = data['interior'][:, 1:2].clone().detach().requires_grad_(requires_grad).to(self.device)

        xb = data['boundary'][:, 0:1].clone().detach().requires_grad_(requires_grad).to(self.device)
        yb = data['boundary'][:, 1:2].clone().detach().requires_grad_(requires_grad).to(self.device)

        f = data['forcing'].clone().detach().requires_grad_(False).to(self.device)
        g = data['boundary_true'].clone().detach().requires_grad_(False).to(self.device)

        return x, y, xb, yb, f, g

    def net_u(self, x):
        z = self.feature_extractor(x)

        if self.bias:
            z = torch.cat((z, torch.ones(z.shape[0], 1, dtype=z.dtype)), dim=1)

        return z @ self.W.t()

    def loss(self, x, y, xb, yb, f, g, solve_W=True):

        # form Laplacian of output features
        lap_z_pred = self.lap_z(x, y)
        z_pred = self.feature_extractor(torch.cat((xb, yb), dim=1))

        # form SlimTik
        if solve_W:
            with torch.no_grad():
                n = f.shape[0]
                nb = g.shape[0]
                nex = n + nb

                beta1 = 1.0
                beta2 = 1.0
                if self.reduction == 'mean':
                    beta1 = 1.0 / math.sqrt(n)
                    beta2 = 1.0 / math.sqrt(nb)

                # concatenate and orient in Matlab code version
                Z = torch.cat((beta1 * lap_z_pred.reshape(n, -1), beta2 * z_pred.reshape(nb, -1)), dim=0)
                Z = Z.transpose(0, 1)
                if self.bias:
                    Z = torch.cat((Z, torch.ones(1, nex, dtype=Z.dtype)), dim=0)

                C = torch.cat((beta1 * f.reshape(-1), beta2 * g.reshape(-1)), dim=0)
                C = C.reshape(1, -1)

                if self.iter > self.memory_depth:
                    self.M = self.M[:, nex:]

                self.M = torch.cat((self.M, Z), dim=1)

                self.solve(Z, C)
                self.iter += 1
                self.alpha = self.sumLambda / (self.iter + 1)
                self.alphaHist += [self.alpha]

        # apply W in pytorch orientation
        m = lap_z_pred.shape[1]
        f_pred = lap_z_pred @ self.W[:, :m].t()
        ub_pred = z_pred @ self.W[:, :m].t()

        if self.bias:
            f_pred = f_pred + self.W[:, -1]
            ub_pred = ub_pred + self.W[:, -1]

        self.loss_u = (0.5 / x.numel()) * torch.norm(f_pred.view(-1) - f.view(-1)) ** 2
        self.loss_b = (0.5 / xb.numel()) * (torch.norm(ub_pred.view(-1) - g.view(-1)) ** 2)

        return self.loss_u + self.loss_b

    def solve(self, Z, C, dtype=torch.float64):

        with torch.no_grad():
            W, info = tiksolve.solve(Z, C, self.M, self.W, self.sumLambda, Lambda=self.Lambda,
                                     dtype=dtype, opt_method=self.opt_method,
                                     lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        self.W = W
        self.sumLambda = info['sumLambda']
        self.Lambda = info['LambdaBest']
        self.LambdaHist += [self.Lambda]

    def lap_z(self, x, y):
        z = self.feature_extractor(torch.cat((x, y), dim=1))

        # loop over output features (TODO: could also loop over samples)
        lap_z = torch.zeros(x.numel(), z.shape[1])
        for i in range(z.shape[1]):

            z_x = torch.autograd.grad(
                z[:, i], x,
                grad_outputs=torch.ones_like(z[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]

            z_y = torch.autograd.grad(
                z[:, i], y,
                grad_outputs=torch.ones_like(z[:, i]),
                retain_graph=True,
                create_graph=True
            )[0]

            z_xx = torch.autograd.grad(
                z_x, x,
                grad_outputs=torch.ones_like(z_x),
                retain_graph=True,
                create_graph=True
            )[0]

            z_yy = torch.autograd.grad(
                z_y, y,
                grad_outputs=torch.ones_like(z_y),
                retain_graph=True,
                create_graph=True
            )[0]

            lap_z[:, i] = (z_xx + z_yy).view(-1)

        return lap_z

    def net_f(self, x, y):
        # form Laplacian
        u = self.net_u(torch.cat((x, y), dim=1))

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        lap_u = u_xx + u_yy

        return lap_u

    def print_outs(self):
        results = {
            'str': ('|W|', 'LastLambda', 'alpha', 'iter', 'memDepth') + ('loss_u', 'loss_b'),
            'frmt': '{:<18.4e}{:<18.4e}{:<18.4e}{:<18d}{:<18d}{:<18.4e}{:<18.4e}',
            'val': None
        }

        results['val'] = [torch.norm(self.W).item(), self.Lambda, self.alpha, self.iter, self.memory_depth] + \
                         [self.loss_u.item(), self.loss_b.item()]

        return results
