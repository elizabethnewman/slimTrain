import torch


def sgcv(Lambda, ZU, WZC, WZCZU, WU, S, sumLambda, n_calTk, n_target):
    with torch.no_grad():
        rnorm = res_norm(Lambda, ZU, WZC, WZCZU, WU, S, sumLambda)
        tterm = trace_term(Lambda, ZU, S, sumLambda, n_target)
        f = rnorm / ((n_calTk - tterm) ** 2)
    return f


def res_norm(Lambda, ZU, WZC, WZCZU, WU, S, sumLambda):
    with torch.no_grad():
        sigma_inv = 1.0 / (S ** 2 + sumLambda + Lambda.view(-1, 1))
        m1 = WZCZU + Lambda.view(Lambda.numel(), 1, 1) * WU.unsqueeze(0)
        m2 = ZU.t().unsqueeze(0) * sigma_inv.view(sigma_inv.shape[0], -1, 1)
        res = torch.matmul(m1, m2) - WZC
        res_norm = torch.reshape(torch.sum(res ** 2, dim=(1, 2)), [-1, 1])
    return res_norm


def trace_term(Lambda, ZU, S, sumLambda, n_target):
    with torch.no_grad():
        sigma_inv = 1.0 / (S ** 2 + sumLambda + torch.reshape(Lambda, [-1, 1])).t()
        sumZU = torch.sum(ZU ** 2, dim=0).unsqueeze_(0)
        tterm = torch.reshape(n_target * (sumZU @ sigma_inv), [-1, 1])

    return tterm
