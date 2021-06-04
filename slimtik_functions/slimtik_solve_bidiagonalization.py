import torch
from slimtik_functions.linear_operators import DenseMatrix, ConcatenatedLinearOperator


def solve_bidiagonalization_sgcv(linOpA, b, linOpM, max_iter, RegParam='gcv',
                                 x_true=None, reorth=True, tik=True, decomp_out=False,
                                 dtype=torch.float64, verbose=False):
    """

    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # setup linear operators
    if isinstance(linOpA, torch.Tensor) and linOpA.ndim == 2:
        linOpA = DenseMatrix(linOpA)

    if isinstance(linOpM, torch.Tensor) and linOpM.ndim == 2:
        linOpM = DenseMatrix(linOpM)

    # TODO: check for compatible datatypes for linear operators

    # ---------------------------------------------------------------------------------------------------------------- #
    # data types
    device_orig = b.device
    dtype_orig = b.dtype
    if dtype is None:
        dtype = b.dtype

    # tolerance
    eps = torch.finfo(dtype).eps

    # ---------------------------------------------------------------------------------------------------------------- #
    # initialize
    b = b.reshape(-1).to(dtype=dtype, device='cpu')
    d = apply_linear_operator_transpose((linOpM, linOpA), b.to(dtype=linOpA.dtype, device=linOpA.device))
    d = d.reshape(-1).to(dtype=dtype, device='cpu')
    n = d.numel()
    m = b.numel()

    x0 = torch.zeros(n, dtype=dtype, device='cpu')
    x = x0
    r = b
    beta = torch.norm(r)
    nrmb = torch.norm(b)

    U = torch.zeros(m, max_iter + 1, dtype=dtype, device='cpu')
    B = torch.zeros(max_iter + 1, max_iter, dtype=dtype, device='cpu')
    V = torch.zeros(n, max_iter, dtype=dtype, device='cpu')
    rhs = torch.zeros(max_iter + 1, dtype=dtype, device='cpu')

    # information to store
    info = {'Xnrm': [], 'Rnrm': [], 'Enrm': [], 'RegParamVect': [], 'StopFlag': None, 'X': None,
            'B': None, 'V': None, 'U': None}
    info['Rnrm'] += [1.0]

    if x_true is not None:
        nrmtrue = torch.norm(x_true)
        info['Enrm'] += [1.0]
        # BestEnrm = 1e10
        # BestReg = {'RegP': None, 'It': None, 'Enrm': None, 'Xnrm': [], 'Rnrm': []}

    # ---------------------------------------------------------------------------------------------------------------- #
    # MAIN CODE
    # ---------------------------------------------------------------------------------------------------------------- #
    u = r
    U[:, 0] = u / beta
    rhs[0] = beta

    k = 0
    while k < max_iter:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # update V
        v = apply_linear_operator_transpose((linOpM, linOpA), U[:,k].to(dtype=linOpA.dtype, device=linOpA.device))
        v = v.reshape(-1).to(dtype=dtype, device='cpu')

        if k > 0:
            v -= beta * V[:, k - 1]

        if reorth:
            v = reorthogonalize(v, V)

        alpha = torch.norm(v)
        V[:, k] = v / alpha

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # update U
        u = apply_linear_operator((linOpM, linOpA), V[:, k].to(dtype=linOpA.dtype, device=linOpA.device))
        u = u.reshape(-1).to(dtype=dtype, device='cpu')
        u -= alpha * U[:, k]

        if reorth:
            u = reorthogonalize(u, U)

        beta = torch.norm(u)
        U[:, k + 1] = u / beta

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # update B and right-hand side
        B[k, k] = alpha
        B[k + 1, k] = beta
        rhsk = rhs[:k + 2]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # check for GKL breakdown
        if torch.abs(alpha) <= eps or torch.abs(beta) <= eps:
            if verbose:
                print('Golub-Kahan bidiagonalization breaks down')
            B = B[:k + 2, :k + 1]
            V = V[:, :k + 1]
            U = U[:, :k + 2]

            info['Xnrm'] = info['Xnrm'][:k]
            info['Rnrm'] = info['Rnrm'][:k]
            info['RegParamVect'] = info['RegParamVect'][:k]
            if x_true is not None:
                info['Enrm'] = info['Enrm'][:k]

            # stop because the bidiagonal matrix is (numerically) singular
            # No choice: we simply cannot compute the solution anymore
            info['StopFlag'] = 'Breakdown of the Golub-Kahan bidiagonalization algorithm'
            break

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # compute svd of the bidiagonal matrix B and update rhs

        # form Bk
        Bk = B[:k + 2, :k + 1]

        # compute full svd
        Uk, Sk, Vk = torch.svd(Bk, some=False)

        # new right-hand side
        rhskhat = Uk.t() @ rhsk

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # choose Tikhonhov regularization parameter
        if tik:
            if isinstance(RegParam, float):
                RegParamk = RegParam
            else:
                raise NotImplementedError
        else:
            # no regularization
            RegParamk = 0

        # store regularizaation parameter
        info['RegParamVect'] += [RegParamk]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # solve!
        Dk = Sk ** 2 + RegParamk ** 2
        rhskhat = Sk * rhskhat[:k + 1]
        y_hat = rhskhat[:k + 1] / Dk
        y = Vk @ y_hat

        info['Rnrm'] += [(torch.norm(rhsk - Bk @ y) / nrmb).item()]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
        # update
        d = V[:, :k + 1] @ y
        x = x0 + d

        info['Xnrm'] += [(torch.norm(x)).item()]
        if x_true is not None:
            info['Enrm'] += [(torch.norm(x.reshape(-1) - x_true.reshape(-1)) / nrmtrue).item()]

        # update iterate
        k += 1

    # ---------------------------------------------------------------------------------------------------------------- #
    # final results
    if k == max_iter:
        info['StopFlag'] = 'reached maximum number of iterations'

    if decomp_out:
        info['U'] = U[:, :k + 2].to(dtype_orig)
        info['B'] = B[:k + 2, :k + 1].to(dtype_orig)
        info['V'] = V[:, :k + 1].to(dtype_orig)

    # recast
    x = x.to(dtype=dtype_orig, device=device_orig)

    return x, info


def reorthogonalize(w, V):
    # orthogonalize w to all columns of V
    for j in range(V.shape[1]):
        w -= torch.dot(V[:, j], w) * V[:, j]

    return w


def apply_linear_operator(linear_operator_list, x):
    # x is a vector that goes into every operator

    b = torch.empty(0, device=x.device)
    for linOp in linear_operator_list:
        b = torch.cat((b, linOp.A(x).view(-1)), dim=0)

    return b


def apply_linear_operator_transpose(linear_operator_list, b):
    # b is a vector with different partitions going into different transpose operators
    numel_in = linear_operator_list[0].numel_in()

    x = torch.zeros(numel_in, device=b.device, dtype=b.dtype)

    count = 0
    for linOp in linear_operator_list:
        n = linOp.numel_out()
        x += linOp.AT(b[count:count + n])
        count += n

    return x.to(b.device)


def resnorm(trial_lambda, B, V, linOpA, w, sumLambda, beta, rhs):
    Ub, Sb, Vb = torch.svd(B, some=True)
    sigma_inv = 1.0 / (Sb ** 2 + sumLambda + trial_lambda)

    e1 = torch.eye(Ub.shape[1])
    y = Vb @ torch.diag(sigma_inv * Sb) @ Ub @ (beta * e1) + Vb @ torch.diag(sigma_inv) @ Vb.T @ (trial_lambda * V.T @ w)
    s = V @ y

    Aw = apply_linear_operator((linOpA,), w.to(dtype=linOpA.dtype, device=linOpA.device))
    Aw = Aw.reshape(-1).to(dtype=rhs.dtype, device=rhs.device)

    As = apply_linear_operator((linOpA,), s.to(dtype=linOpA.dtype, device=linOpA.device))
    As = As.reshape(-1).to(dtype=rhs.dtype, device=rhs.device)

    r = Aw - rhs
    return torch.norm(r - As)
