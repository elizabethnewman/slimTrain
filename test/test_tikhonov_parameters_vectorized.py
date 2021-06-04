import torch
from slimtik_functions import slimtik_solve as tiksolvevec

torch.manual_seed(20)
torch.set_default_dtype(torch.float64)

n_ex = 11
n_target = 3
n_calTk = 10
n_out = 5
r = 2
sumLambda = 1e-4
Lambda = torch.tensor([0, 1e-2, 1.0, 1.2])

with torch.no_grad():
    I = torch.eye(n_target)

    # problem setup
    A = torch.randn(n_ex * n_target, n_out * n_target)
    M = torch.randn(r * n_ex * n_target, n_out * n_target)
    M = torch.cat((M, A), dim=0)

    c = torch.randn(A.shape[0])
    w = torch.randn(n_target * n_out)

    # pre-computations
    _, S, V = torch.svd(M)
    Awc = A @ w - c
    AV = A @ V
    AVTAwc = AV.t() @ Awc
    VTw = V.t() @ w

    # test trace term
    tterm = tiksolvevec.trace_term(Lambda, AV, S, sumLambda, n_target)

    # true trace term
    tterm_true = torch.zeros(Lambda.numel())
    for i in range(Lambda.numel()):
        Tk = (Lambda[i] + sumLambda) * torch.eye(M.shape[1]) + M.t() @ M
        tterm_true[i] = torch.trace(A @ torch.pinverse(Tk) @ A.t())

    print(tterm.view(-1) - tterm_true.view(-1))

    # test residual term
    rnorm = tiksolvevec.res_norm(Lambda, AV, Awc, AVTAwc, VTw, S, sumLambda)

    # true residual norm
    rnorm_true = torch.zeros(Lambda.numel())
    for i in range(Lambda.numel()):
        Tk = (Lambda[i] + sumLambda) * torch.eye(M.shape[1]) + M.t() @ M
        res = A @ w - c
        s = torch.pinverse(Tk) @ (A.t() @ res + Lambda[i] * w)
        rnorm_true[i] = torch.norm(A @ (w - s) - c) ** 2

    print(rnorm.view(-1) - rnorm_true.view(-1))






