import torch
from utils import parameters_norm, grad_norm
import math


class PoissonPINN:

    def __init__(self, net, device='cpu'):
        super(PoissonPINN, self).__init__()

        self.net_u = net
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
