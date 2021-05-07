
import torch
from slimtik_functions.linear_operators import DenseMatrix
import numpy as np
from math import sqrt, log10


def lanczos_bidiagonalization(linOp, b, max_dim):

    # transform a matirx into a linear operator
    if isinstance(linOp, torch.Tensor) and linOp.ndim == 2:
        linOp = DenseMatrix(linOp)

    m = b.numel()
    b_shape = b.shape
    beta = torch.norm(b)
    u = b / beta
    ATu = linOp.AT(u)
    v_shape = ATu.shape
    ATu = ATu.reshape(-1)

    n = ATu.numel()
    v = torch.zeros(n)

    U = torch.zeros(m, max_dim + 1)
    V = torch.zeros(n, max_dim)
    B = torch.zeros(max_dim, 2)

    U[:, 0] = u.view(-1)

    k = 0
    while k < max_dim:
        if k > 0:
            ATu = linOp.AT(u.view(b_shape)).reshape(-1)

        r = ATu - beta * v

        # reorthogonalization
        for j in range(k):
            r -= torch.dot(V[:, j], r) * V[:, j]

        alpha = torch.norm(r)
        v = r / alpha

        B[k, 1] = alpha
        V[:, k] = v

        Av = linOp.A(v.reshape(v_shape)).reshape(-1)
        b = Av - alpha * u.reshape(-1)

        # reorthogonalization
        for j in range(k - 1):
            b -= torch.dot(U[:, j], b) * U[:, j]

        beta = torch.norm(b)
        u = b / beta

        B[k, 0] = beta
        U[:, k + 1] = u.reshape(-1)
        k += 1

    # TODO: make B sparse
    idx = torch.arange(k)
    B2 = torch.zeros(k + 1, k)
    B2[idx + 1, idx] = B[:, 0]
    B2[idx, idx] = B[:, 1]

    return U, B2, V


# https://github.com/jnagy1/IRtools/blob/master/IRcodes/IRhybrid_lsqr.m
def hybrid_lsqr_gcv(linOp, b, max_iter, RegParam='gcv', x_true=None, reorth=True, tik=True, dtype=torch.float64,
                    verbose=False, decomp_out=False):
    # TODO: make sure to cast data correctly # TODO: make sure to cast data correctly
    if isinstance(linOp, torch.Tensor) and linOp.ndim == 2:
        linOp = DenseMatrix(linOp)

    # ---------------------------------------------------------------------------------------------------------------- #
    # setup
    eps = torch.finfo(dtype).eps

    # initialize
    b = b.reshape(-1)
    d = linOp.AT(b).reshape(-1)
    n = d.numel()
    m = b.numel()

    x0 = torch.zeros(n).to(b.dtype)
    x = x0
    r = b
    beta = torch.norm(r)
    nrmb = torch.norm(b)

    U = torch.zeros(m, max_iter + 1)
    B = torch.zeros(max_iter + 1, max_iter)
    V = torch.zeros(n, max_iter)
    rhs = torch.zeros(max_iter + 1)

    # information to store
    info = {'Xnrm': [], 'Rnrm': [], 'Enrm': [], 'RegParamVect': [], 'StopFlag': None, 'X': None,
            'B': None, 'V': None, 'U': None}

    if x_true is not None:
        nrmtrue = torch.norm(x_true)
        # BestEnrm = 1e10
        # BestReg = {'RegP': None, 'It': None, 'Enrm': None, 'Xnrm': [], 'Rnrm': []}

    # ---------------------------------------------------------------------------------------------------------------- #
    # main code
    u = r
    U[:, 0] = u / beta
    rhs[0] = beta

    k = 0
    while k < max_iter:
        w = linOp.AT(U[:, k]).reshape(-1)

        if k > 0:
            w -= beta * V[:, k - 1]

        if reorth:
            for jj in range(k - 1):
                w - torch.dot(V[:, jj], w) * V[:, jj]

        alpha = torch.norm(w)
        V[:, k] = w / alpha

        u = linOp.A(V[:, k]).reshape(-1)
        u -= alpha * U[:, k]
        if reorth:
            for jj in range(k - 1):
                u -= torch.dot(U[:, jj], u) * U[:, jj]

        beta = torch.norm(u)
        U[:, k + 1] = u / beta
        B[k, k] = alpha
        B[k + 1, k] = beta
        rhsk = rhs[:k + 2]

        # breakdown
        if torch.abs(alpha) <= eps or torch.abs(beta) <= eps:
            if verbose:
                print('Golub-Kahan bidiagonalization breaks down')
            B = B[:k + 2, :k + 1]
            V = V[:, :k + 1]
            U = U[:, :k + 2]

            if k > 0:
                info['Xnrm'] = info['Xnrm'][:k]
                info['Rnrm'] = info['Rnrm'][:k]
                info['RegParamVect'] = info['RegParamVect'][:k]
                if x_true is not None:
                    info['Enrm'] = info['Enrm'][:k]

                # stop because the bidiagonal matrix is (numerically) singular
                # No choice: we simply cannot compute the solution anymore
                info['StopFlag'] = 'Breakdown of the Golub-Kahan bidiagonalization algorithm'
                break

        # form Bk
        Bk = B[:k + 2, :k + 1]

        # compute full svd
        Uk, Sk, Vk = torch.svd(Bk, some=False)

        # new right-hand side
        rhskhat = Uk.t() @ rhsk

        if tik:
            if isinstance(RegParam, float):
                RegParamk = RegParam
                info['RegParamVect'] += [RegParamk]
            else:
                # use sgcv
                alpha = torch.logspace(log10(eps), 3, 100)
                trial_points = gcv_trial_points(alpha, Uk[0, :], Sk, nrmb)
                RegParamk = torch.min(trial_points)
                info['RegParamVect'] += [RegParamk]
        else:
            # no regularization
            RegParamk = 0


        # solve!
        Dk = Sk ** 2 + RegParamk ** 2
        rhskhat = Sk * rhskhat[:k + 1]
        y_hat = rhskhat[:k + 1] / Dk
        y = Vk @ y_hat
        info['Rnrm'] = torch.norm(rhsk - Bk @ y) / nrmb

        # update
        d = V[:, :k + 1] @ y
        x = x0 + d
        info['Xnrm'] += [torch.norm(x)]

        if x_true is not None:
            info['Enrm'] += [torch.norm(x.reshape(-1) - x_true.reshape(-1)) / nrmtrue]

        # update iterate
        k += 1

    if k == max_iter:
        info['StopFlag'] = 'reached maximum number of iterations'

    if decomp_out:
        info['U'] = U[:, :k + 2]
        info['B'] = B[:k + 2, :k + 1]
        info['V'] = V[:, :k + 1]

    return x, info



def gcv_trial_points(alpha, u, s, beta):
    # alpha could be many trial points

    k = s.numel()
    beta2 = beta ** 2
    alpha2 = (alpha ** 2).reshape(-1, 1)
    s2 = (s ** 2).reshape(1, -1)

    t1 = alpha2 / (s2 + alpha2)

    num = (k * beta2) * (torch.sum((u[:k] * t1) ** 2, dim=1) + u[k] ** 2)
    den = (1 + torch.sum(t1, dim=1)) ** 2

    return num / den



