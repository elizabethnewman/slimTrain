import torch
import math


def solve(Z, C, M, W, sumLambda,
          dtype=torch.float64, opt_method=None,
          lower_bound=1e-7, upper_bound=1e-3, Lambda=1.0):
    """
    Solve ||S[M, Z,

    Note that M should contain Z as well
    """

    orig_dtype = Z.dtype
    U, S, _ = torch.svd(M.to(dtype))

    Z = Z.to(dtype)
    C = C.to(dtype)
    W = W.to(dtype)
    WZC = W @ Z - C

    if opt_method == 'trial_points':
        ZU = Z.t() @ U
        WZCZU = WZC @ ZU
        WU = W @ U

        n_target, n_calTk = C.shape

        # choose candidates
        Lambda = choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=dtype)

        # approximate minimum of gcv function
        f = sgcv(Lambda, ZU, WZC, WZCZU, WU, S, sumLambda, n_calTk, n_target)
        idx = torch.argmin(f, dim=0)
        Lambda_best = Lambda[idx].item()
    else:
        Lambda_best = Lambda

    # make sure Lambda_best is feasible
    if sumLambda + Lambda_best <= 0:
        raise ValueError('sumLambda must be positive!')

    # update sumLambda
    sumLambda += Lambda_best

    # update W using Sherman-Morrison-Woodbury
    alpha = 1.0 / sumLambda
    I = torch.eye(W.shape[1])
    s2 = S ** 2
    Binv = alpha * I - alpha ** 2 * ((U @ torch.diag(s2 / (1.0 + alpha * s2))) @ U.t())
    W -= (WZC @ Z.t() + Lambda_best * W) @ Binv

    # return useful information
    info = {'sumLambda': sumLambda, 'LambdaBest': Lambda_best, 'Rnrm': (torch.norm(W @ Z - C) / torch.norm(C)).item()}

    return W.to(orig_dtype), info


def choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=torch.float64):
    eps = torch.finfo(dtype).eps

    # Lambda can be as large as the upper bound
    Lambda1 = torch.logspace(math.log10(eps), math.log10(upper_bound), 100)

    n_low = sumLambda / 2 - lower_bound  # divide by 2 to avoid numerical issues
    if n_low <= 0:
        # sumLambda is already less than or equal to lower_bound
        # don't decrease regularization parameter
        Lambda2 = torch.empty(0)
    else:
        Lambda2 = -torch.logspace(math.log10(n_low), math.log10(eps), 100)

    Lambda = torch.cat((Lambda2, Lambda1), dim=0)

    return Lambda


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