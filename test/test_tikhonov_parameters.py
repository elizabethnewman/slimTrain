import torch
from old_code.slimtik_functions import tikhonov_parameters as tp

torch.manual_seed(20)
torch.set_default_dtype(torch.float64)

n_target = 3
n_calTk = 10
n_out = 5
r = 2
sumLambda = 1e-4
Lambda = torch.tensor([0, 1e-2, 1.0, 1.2])

with torch.no_grad():
    I = torch.eye(n_target)

    # problem setup
    M = torch.empty(0)
    B = torch.empty(0)
    A = torch.empty(0)
    for i in range(r + 1):
        Z = torch.randn(n_out, n_calTk)
        C = torch.randn(n_target, n_calTk)
        M = torch.cat((M, Z), dim=1)
        B = torch.cat((B, C), dim=1)
        A = torch.cat((A, torch.kron(Z.t().contiguous(), I)), dim=0)

    b = B.t().contiguous().view(-1)
    W = torch.randn(n_target, n_out)

    # test trace term
    Zk = M[:, -n_calTk:]
    Ck = C[:, -n_calTk:]
    U, S, _ = torch.svd(M)
    ZU = Z.t() @ U
    WZC = W @ Z - C
    WZCZU = WZC @ ZU
    WU = W @ U
    tterm = tp.trace_term(Lambda, ZU, S, sumLambda, n_target)

    # truth
    Ak = A[(-n_calTk * (r + 1)):, :]
    bk = b[(-n_calTk * (r + 1)):]

    tterm_true = torch.zeros(Lambda.numel())
    for i in range(Lambda.numel()):
        Tk = (Lambda[i] + sumLambda) * torch.eye(A.shape[1]) + A.t() @ A
        tterm_true[i] = torch.trace(Ak @ torch.inverse(Tk) @ Ak.t())

    print(tterm.view(-1) - tterm_true.view(-1))

    # test residual term
    rnorm = tp.res_norm(Lambda, ZU, WZC, WZCZU, WU, S, sumLambda)

    w = W.t().contiguous().view(-1)
    rnorm_true = torch.zeros(Lambda.numel())
    rnorm_true2 = torch.zeros(Lambda.numel())
    for i in range(Lambda.numel()):
        tmp1 = A.t() @ A + (Lambda[i] + sumLambda) * torch.eye(A.shape[1])
        tmp2 = Ak.t() @ (Ak @ w - bk) + Lambda[i] * w
        s = torch.inverse(tmp1) @ tmp2
        wLam = w - s

        mm = M @ M.t() + (Lambda[i] + sumLambda) * torch.eye(n_out)
        S = ((W @ Zk - Ck) @ Zk.t() + Lambda[i] * W) @ torch.inverse(mm)
        WLam = W - S

        rnorm_true[i] = torch.norm(Ak @ wLam - bk) ** 2
        rnorm_true2[i] = torch.norm(WLam @ Zk - Ck) ** 2

    print(rnorm.view(-1) - rnorm_true.view(-1))






