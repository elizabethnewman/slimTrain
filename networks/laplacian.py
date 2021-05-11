import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math


class AntiTanhWithSecondDerivative(nn.Module):

    def __init__(self):
        super(AntiTanhWithSecondDerivative, self).__init__()

    def f(self, x):
        return torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

    def df(self, x):
        return torch.tanh(x)

    def d2f(self, x):
        return 1 - torch.tanh(x) ** 2



class SingleLayer:
    """
    Single layer for neural networks
    f(x,theta) = act( K * x + b), theta = (K,b)
    """

    def __init__(self, in_features, out_features, bias=True, act=AntiTanhWithSecondDerivative()):
        super(SingleLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.act = act

        self.K = torch.randn(out_features, in_features)

        if bias is True:
            self.b = torch.randn(out_features, 1)

        self.store_intermediate = None

    def forward(self, x):
        """
        :param x:
        :param K:
        :param b:
        :return:
        """
        z = torch.mm(self.K, x)

        if self.bias:
            z += self.b

        self.store_intermediate = (x, z)
        a = self.act.f(z)
        return a

    def vJacw(self, dK, db, w):
        """
        Compute  w' * J_theta f(x,K,b) * (dK,db),
        where dK and db are perturbations of the weights and w is a given vector in R^m.
        :param x:
        :param K:
        :param b:
        :param dK:
        :param db:
        :param w:
        :return: \nabla_x (w'*f(x,theta)) and the above matvec
        """

        x, z = self.store_intermediate

        da = self.act.df(z)
        dx = torch.mm(self.K.transpose(0, 1), da * w)
        vJw = torch.sum((w * da) * (torch.mm(dK, x) + db), dim=0, keepdim=True)
        return dx, vJw

    def JacTw(self, w):
        """
        Compute  w' * J_theta f(x,K,b),
        :param x:
        :param K:
        :param b:
        :param dK:
        :param db:
        :param w:
        :return: \nabla_x (w'*f(x,theta))
        """
        _, z = self.store_intermediate

        da = self.act.df(z)
        dx = torch.mm(self.K.transpose(0, 1), da * w)
        return dx

    def LapAndJac(self, w, Jac=None):
        """
        Compute Laplacian of x \mapsto w'*f(x,theta) and
        the Jacobian Jac_x f(x,theta).
        :param x:
        :param K:
        :param b:
        :param w:
        :param Jac: flag to compute the Jacobian (default = None)
        :return:
        """
        _, z = self.store_intermediate
        n = z.shape[1]

        if Jac is not None:
            Jac = torch.mm(self.K, Jac.reshape(self.K.shape[1], -1)).reshape(self.K.shape[1],-1,n)
        else:
            Jac = self.K.unsqueeze(2)

        Lap = torch.sum((w * self.act.d2f(z)).reshape(self.K.shape[0], -1, n) * torch.pow(Jac, 2), dim=(0, 1))
        Jac = self.act.df(z).unsqueeze(1) * Jac

        return Lap.reshape(-1, n), Jac
