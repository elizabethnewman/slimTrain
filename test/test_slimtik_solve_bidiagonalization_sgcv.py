from slimtik_functions.linear_operators import *

torch.set_default_dtype(torch.float64)
torch.manual_seed(20)

# ==================================================================================================================== #
print('Dense')
m = 10
n = 20

A = torch.randn(m, n)
linOpA = DenseMatrix(A)

M = torch.randn(4 * m, n)
linOpM = DenseMatrix(M)

