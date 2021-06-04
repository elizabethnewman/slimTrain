import torch
import math


def solve(A, c, M, w, sumLambda, n_calTk, n_target,
          dtype=torch.float64, device='cpu', opt_method=None,
          lower_bound=1e-7, upper_bound=1e-3, Lambda=1.0):
    """
    Solve ||S[M, Z,

    Note that M should contain Z as well
    """

    orig_dtype = A.dtype
    orig_device = A.device
    _, S, V = torch.svd(M.to(dtype))
    A = A.to(dtype=dtype, device=device)
    c = c.to(dtype=dtype, device=device).reshape(-1, 1)
    w = w.to(dtype=dtype, device=device).reshape(-1, 1)

    Awc = A @ w - c

    if opt_method == 'trial_points':
        AV = A @ V
        AVTAwc = AV.t() @ Awc
        VTw = V.t() @ w

        # choose candidates
        Lambda = choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=dtype)

        # approximate minimum of gcv function
        f = sgcv(Lambda, AV, Awc, AVTAwc, VTw, S, sumLambda, n_calTk, n_target)
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
    # alpha = 1.0 / sumLambda
    I = torch.eye(w.numel())
    s2 = S ** 2
    w -= (V @ torch.diag(1 / s2) @ V.t()) @ (A.t() @ Awc + Lambda_best * w)

    # return useful information
    info = {'sumLambda': sumLambda, 'LambdaBest': Lambda_best, 'Rnrm': (torch.norm(A @ w - c) / torch.norm(c)).item()}

    return w.to(dtype=orig_dtype, device=orig_device), info


def choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=torch.float64):
    eps = torch.finfo(dtype).eps

    # Lambda can be as large as the upper bound
    Lambda1 = torch.logspace(math.log10(eps), math.log10(upper_bound), 30)

    n_low = sumLambda / 2 - lower_bound  # divide by 2 to avoid numerical issues
    if n_low <= 0:
        # sumLambda is already less than or equal to lower_bound
        # don't decrease regularization parameter
        Lambda2 = torch.empty(0)
    else:
        Lambda2 = -torch.logspace(math.log10(n_low), math.log10(eps), 30)

    Lambda = torch.cat((Lambda2, Lambda1), dim=0)

    return Lambda


def sgcv(Lambda, AV, Awc, AVTAwc, VTw, S, sumLambda, n_calTk, n_target):
    with torch.no_grad():
        rnorm = res_norm(Lambda, AV, Awc, AVTAwc, VTw, S, sumLambda)
        tterm = trace_term(Lambda, AV, S, sumLambda, n_target)
        f = rnorm / ((n_calTk - tterm) ** 2)
    return f


def res_norm(Lambda, AV, Awc, AVTAwc, VTw, S, sumLambda):
    with torch.no_grad():
        sigma_inv = 1.0 / (S ** 2 + sumLambda + Lambda.view(-1, 1))
        m1 = AVTAwc.reshape(-1) + Lambda.view(Lambda.numel(), 1, 1) * VTw.reshape(-1).unsqueeze(0)
        m2 = AV.t().unsqueeze(0) * sigma_inv.view(sigma_inv.shape[0], -1, 1)
        tmp = torch.matmul(m1, m2)
        res = tmp - torch.kron(Awc.reshape(1, -1), torch.ones(tmp.shape[1]).reshape(-1, 1))
        res_norm = torch.reshape(torch.sum(res ** 2, dim=(1, 2)), [-1, 1])
        # res_norm = torch.norm(res, dim=(1, 2)) ** 2
    return res_norm.reshape(-1, 1)


def trace_term(Lambda, AV, S, sumLambda, n_target):
    with torch.no_grad():
        sigma_inv = 1.0 / (S ** 2 + sumLambda + torch.reshape(Lambda, [-1, 1])).t()
        sumAV = torch.sum(AV ** 2, dim=0).unsqueeze_(0)
        tterm = torch.reshape(sumAV @ sigma_inv, [-1, 1])

    return tterm
