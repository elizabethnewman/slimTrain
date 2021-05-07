
import torch
import time
from utils import optimizer_params, parameters_norm, grad_norm
import math


def train_lbfgs(pinn, optimizer, data, max_iter, verbose=True):

    opt_params = optimizer_params(optimizer, spacing=18)
    net_params = pinn.print_outs()

    results = {
        'str': ('iter', 'pinn_iter',) + opt_params['str'] + ('|param|', '|g| / |g0|') + net_params['str'] + ('loss',),
        'frmt': '{:<18d}{:<18d}' + opt_params['frmt'] + '{:<18.4e}{:<18.4e}' + net_params['frmt'] + '{:<18.4e}',
        'val': None
    }
    results['val'] = torch.empty(0)

    if verbose:
        print(('{:<18s}' * len(results['str'])).format(*results['str']))

    # extract data
    training_data = pinn.extract_data(data, requires_grad=True)

    # initial evaluation
    loss = pinn.loss(*training_data)
    loss.backward()
    p_norm0 = parameters_norm(pinn.net_u)
    g_norm0 = grad_norm(pinn.net_u)

    his = len(results['str']) * [0]
    his[0] = -1
    his[-5:] = [p_norm0, g_norm0 / g_norm0, pinn.loss_u.item(), pinn.loss_b.item(), loss.item()]
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(results['frmt'].format(*his))

    for iter in range(max_iter):

        def closure():
            optimizer.zero_grad()
            loss = pinn.loss(*training_data)
            loss.backward()
            return loss

        optimizer.step(closure)

        loss = pinn.loss(*training_data)
        loss.backward()

        opt_params = optimizer_params(optimizer)
        net_params = pinn.print_outs()
        his = [iter, pinn.iter]
        his += opt_params['val']

        p_norm = parameters_norm(pinn.net_u)
        g_norm = grad_norm(pinn.net_u)
        his += [p_norm, g_norm / g_norm0]
        his += net_params['val']
        his += [loss.item()]

        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose:
            print(results['frmt'].format(*his))

    return results
