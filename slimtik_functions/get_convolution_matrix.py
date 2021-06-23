
import torch
from slimtik_functions.linear_operators import ConcatenatedLinearOperator, LinearOperator
import torch.nn.functional as F

def get_ConcatenatedConv2DTranspose_matrix(linOp: ConcatenatedLinearOperator):

    A_mat = torch.empty(0)
    for lop in linOp.linOpList:
        A = get_Conv2DTranspose_matrix(lop)
        A_mat = torch.cat((A_mat, A), dim=0)

    return A_mat


def get_Conv2DTranspose_matrix(A, shape=None):

    if isinstance(A, ConcatenatedLinearOperator) or isinstance(A, LinearOperator):
        A_mat = get_Conv2DTranspose_matrix_from_linOp(A)
    else:
        # A is data

        C_in, C_out, kH, kW = shape
        if C_out > 1:
            raise ValueError('get_Conv2DTranspose_matrix only accepts 1 output channel')
        N
        A_mat = torch.zeros(kH, kW, N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])
        for i in range(kH):
            for j in range(kW):
                ei = torch.zeros(1, C_out, kH, kW)
                ei[:, :, i, j] = 1.0
                ei = ei.reshape(-1)
                ei = torch.cat((ei, torch.zeros(1)))  # because we have 1 output channel

                Aei = F.conv_transpose2d(A, ei)
                A_mat[i, j] = Aei.reshape(N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])

        linOp.shape_in = (C_in, C_out, kH, kW)
        linOp.data = d.reshape(d_shape)

    return A_mat


def get_Conv2DTranspose_matrix_from_linOp(linOp):
    C_in, C_out, kH, kW = linOp.shape_in

    N = linOp.data.shape[0]

    if C_out > 1:
        raise ValueError('get_Conv2DTranspose_matrix only accepts 1 output channel')

    d = linOp.data
    d_shape = d.shape
    linOp.data = d.reshape(-1, 1, d.shape[2], d.shape[3])
    linOp.shape_in = (1, C_out, kH, kW)
    A_mat = torch.zeros(kH, kW, N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])
    for i in range(kH):
        for j in range(kW):
            ei = torch.zeros(1, C_out, kH, kW)
            ei[:, :, i, j] = 1.0
            ei = ei.reshape(-1)
            ei = torch.cat((ei, torch.zeros(1)))  # because we have 1 output channel

            Aei = linOp.A(ei)
            A_mat[i, j] = Aei.reshape(N, C_in, C_out, linOp.shape_out[1], linOp.shape_out[2])

    linOp.shape_in = (C_in, C_out, kH, kW)
    linOp.data = d.reshape(d_shape)

    # reshape
    A_mat = A_mat.permute(3, 2, 5, 6, 4, 0, 1)
    A_mat = A_mat.reshape(C_in, -1, kH * kW)
    A_mat = list_squeeze(list(torch.tensor_split(A_mat, A_mat.shape[0])))
    A_mat = torch.cat(A_mat, dim=1)

    # add bias
    A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0], 1)), dim=1)

    return A_mat


def list_squeeze(input):

    for i in range(len(input)):
        input[i] = input[i].squeeze()
    return input
