import torch
import math


def solve(A, c, MtM, w, sumLambda, n_calTk, n_target,
          dtype=torch.float64, device='cpu', opt_method=None,
          lower_bound=1e-7, upper_bound=1e-3, Lambda=1.0):
    """
    Solve ||S[M, Z,

    Note that M should contain Z as well
    """

    # original datatype and device
    orig_dtype = A.dtype
    orig_device = A.device

    # mapping to device
    A = A.to(dtype=dtype, device=device)
    c = c.to(dtype=dtype, device=device).reshape(-1, 1)
    w = w.to(dtype=dtype, device=device).reshape(-1, 1)

    # initial residual
    Awc = A @ w - c

    # compute svd for efficient inversion
    _, S2, V = torch.svd(MtM.to(dtype=dtype, device=device))

    if opt_method == 'trial_points':
        # pre-computed variables
        AV = A @ V
        AVTAwc = AV.t() @ Awc
        VTw = V.t() @ w

        # choose candidates
        Lambda = choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=dtype, device=device)

        # approximate minimum of gcv function
        f = sgcv(Lambda, AV, Awc, AVTAwc, VTw, S2, sumLambda, n_calTk, n_target)
        idx = torch.argmin(f, dim=0)
        Lambda_best = Lambda[idx].item()
    else:
        Lambda_best = Lambda

    # make sure Lambda_best is feasible
    if sumLambda + Lambda_best < 0:
        raise ValueError('sumLambda must be nonnegative!')

    if Lambda_best + sumLambda < lower_bound:
        Lambda_best = -sumLambda + lower_bound

    # update sumLambda
    sumLambda += Lambda_best

    # update w
    s2 = S2 + sumLambda
    VTrhs = V.t() @ (A.t() @ Awc + Lambda_best * w)
    s = V @ (VTrhs.reshape(-1) / s2.reshape(-1))
    # s = V @ torch.diag(1 / s2) @ VTrhs
    w -= s.reshape(w.shape)

    # return useful information
    info = {'sumLambda': sumLambda, 'LambdaBest': Lambda_best, 'Rnrm': (torch.norm(A @ w - c) / torch.norm(c)).item()}

    return w.to(dtype=orig_dtype, device=orig_device), info


def choose_Lambda_candidates(sumLambda, upper_bound, lower_bound, dtype=torch.float64, device='cpu'):
    eps = torch.finfo(dtype).eps

    # Lambda can be as large as the upper bound
    Lambda1 = torch.logspace(math.log10(eps), math.log10(upper_bound), 30, device=device).reshape(-1, 1)
    Lambda = torch.cat((-torch.fliplr(Lambda1), Lambda1), dim=0)

    # n_low = sumLambda / 2 - lower_bound  # divide by 2 to avoid numerical issues
    # if n_low <= 0:
    #     # sumLambda is already less than or equal to lower_bound
    #     # don't decrease regularization parameter
    #     Lambda2 = torch.empty(0, device=device)
    # else:
    #     Lambda2 = -torch.logspace(math.log10(n_low), math.log10(eps), 30, device=device)

    # Lambda = torch.cat((Lambda2, Lambda1), dim=0)

    # p = min(math.log10(sumLambda / 2), upper_bound)
    # Lambda = torch.logspace(min(p, -10), max(p, -10), 30, device=device, dtype=dtype).reshape(-1, 1)
    # Lambda = torch.cat((-torch.fliplr(Lambda), Lambda))

    return Lambda.reshape(-1)


def sgcv(Lambda, AV, Awc, AVTAwc, VTw, S2, sumLambda, n_calTk, n_target):
    rnorm = res_norm(Lambda, AV, Awc, AVTAwc, VTw, S2, sumLambda)
    tterm = trace_term(Lambda, AV, S2, sumLambda, n_target)
    f = rnorm / ((n_calTk - tterm) ** 2)
    return f


def res_norm(Lambda, AV, Awc, AVTAwc, VTw, S2, sumLambda):
    sigma_inv = 1.0 / (S2 + sumLambda + Lambda.view(-1, 1))
    m1 = AVTAwc.reshape(-1) + Lambda.view(Lambda.numel(), 1, 1) * VTw.reshape(-1).unsqueeze(0)
    m2 = AV.t().unsqueeze(0) * sigma_inv.view(sigma_inv.shape[0], -1, 1)
    tmp = torch.matmul(m1, m2)
    # res = tmp - torch.kron(Awc.reshape(1, -1), torch.ones(tmp.shape[1]).reshape(-1, 1))
    # res_norm = torch.reshape(torch.sum(res ** 2, dim=(1, 2)), [-1, 1])
    res = tmp - Awc.reshape(1, -1)
    res_norm = torch.norm(res, dim=(1, 2)) ** 2

    return res_norm.reshape(-1, 1)


def trace_term(Lambda, AV, S2, sumLambda, n_target):
    sigma_inv = 1.0 / (S2 + sumLambda + torch.reshape(Lambda, [-1, 1])).t()
    sumAV = torch.sum(AV ** 2, dim=0).unsqueeze_(0)
    tterm = torch.reshape(sumAV @ sigma_inv, [-1, 1])

    return tterm
