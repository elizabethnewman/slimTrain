
import torch
import time
from utils import optimizer_params, parameters_norm, grad_norm
import math
from pinns.poisson import PoissonPINNSlimTik


def train_lbfgs(pinn, optimizer, data, max_iter, verbose=True):

    opt_params = optimizer_params(optimizer, spacing=18)
    net_params = pinn.print_outs()

    results = {
        'str': ('iter', 'pinn_iter',) + opt_params['str'] + ('time', '|param|', '|g| / |g0|')
               + net_params['str'] + ('loss',),
        'frmt': '{:<18d}{:<18d}' + opt_params['frmt'] + '{:<18.2f}{:<18.4e}{:<18.4e}' + net_params['frmt'] + '{:<18.4e}',
        'val': None
    }
    results['val'] = torch.empty(0)

    # extract data
    training_data = pinn.extract_data(data, requires_grad=True)

    # initial evaluation
    loss = pinn.loss(*training_data)
    loss.backward()
    p_norm0 = parameters_norm(pinn.feature_extractor)
    g_norm0 = grad_norm(pinn.feature_extractor)

    his = len(results['str']) * [0]
    his[0] = -1
    his[-5:] = [p_norm0, g_norm0 / g_norm0, pinn.loss_u.item(), pinn.loss_b.item(), loss.item()]
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(('{:<18s}' * len(results['str'])).format(*results['str']))
        print(results['frmt'].format(*his))

    total_start = time.time()
    for iter in range(max_iter):

        def closure():
            optimizer.zero_grad()
            loss = pinn.loss(*training_data)
            loss.backward()
            return loss
        start = time.time()
        optimizer.step(closure)
        end = time.time()

        loss = pinn.loss(*training_data)
        loss.backward()

        opt_params = optimizer_params(optimizer)
        net_params = pinn.print_outs()
        his = [iter, pinn.iter]
        his += opt_params['val']

        p_norm = parameters_norm(pinn.feature_extractor)
        g_norm = grad_norm(pinn.feature_extractor)
        his += [end - start, p_norm, g_norm / g_norm0]
        his += net_params['val']
        his += [loss.item()]

        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose:
            print(results['frmt'].format(*his))

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results


def train_sgd(pinn, optimizer, scheduler, data, num_epochs=5, batch_size=10, log_interval=50, verbose=True):

    opt_params = optimizer_params(optimizer, spacing=18)
    net_params = pinn.print_outs()

    results = {
        'str': ('epoch', 'pinn_iter') + opt_params['str'] + ('time', '|params|')
               + ('running_loss_u', 'running_loss_b', 'running_loss') + net_params['str'] + ('loss',),
        'frmt': '{:<18d}{:<18d}' + opt_params['frmt'] + '{:<18.2f}{:<18.4e}'
                + '{:<18.4e}{:<18.4e}{:<18.4e}' + net_params['frmt'] + '{:<18.4e}',
        'val': None}
    results['val'] = torch.empty(0, len(results['str']))

    # extract data
    training_data = pinn.extract_data(data, requires_grad=True)

    # initial evaluation
    if isinstance(pinn, PoissonPINNSlimTik):
        loss = pinn.loss(*training_data, solve_W=False)
    else:
        loss = pinn.loss(*training_data)

    p_norm0 = parameters_norm(pinn.feature_extractor)
    loss.backward()

    his = len(results['str']) * [0]
    his[0:2] = [-1, pinn.iter]
    his[-7:] = [p_norm0, 0, 0, 0, pinn.loss_u.item(), pinn.loss_b.item(), loss.item()]
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(('{:<18s}' * len(results['str'])).format(*results['str']))
        print(results['frmt'].format(*his))

    total_start = time.time()

    for epoch in range(num_epochs):
        start = time.time()
        train_out = train_one_epoch(pinn, optimizer, data, batch_size=batch_size)
        end = time.time()

        if isinstance(pinn, PoissonPINNSlimTik):
            loss = pinn.loss(*training_data, solve_W=False)
        else:
            loss = pinn.loss(*training_data)

        loss.backward()

        opt_params = optimizer_params(optimizer)
        net_params = pinn.print_outs()

        his = [epoch, pinn.iter]
        his += opt_params['val']

        p_norm = parameters_norm(pinn.feature_extractor)

        his += [end - start, p_norm]
        his += [*train_out]
        his += net_params['val']
        his += [loss.item()]

        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose and not epoch % log_interval:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results


def train_one_epoch(pinn, optimizer, data, batch_size=10):
    pinn.feature_extractor.train()

    running_loss_u = 0
    running_loss_b = 0
    running_loss = 0
    num_samples_u = 0
    num_samples_b = 0

    # extract data
    x, y, xb, yb, f, ub = pinn.extract_data(data, requires_grad=False)

    # shuffle
    n = x.shape[0]
    nb = xb.shape[0]

    batch_idx = torch.randperm(n)
    x, y, f = x[batch_idx], y[batch_idx], f[batch_idx]

    batch_idx_b = torch.randperm(nb)
    xb, yb, ub = xb[batch_idx_b], yb[batch_idx_b], ub[batch_idx_b]

    # form same number of batches for interior and boundary
    num_batches = n // batch_size
    batch_size_b = nb // num_batches

    if batch_size_b == 0:
        raise ValueError('Need more boundary points or fewer batches')

    for i in range(num_batches):
        x_batch = x[i * batch_size:(i + 1) * batch_size].requires_grad_(True)
        y_batch = y[i * batch_size:(i + 1) * batch_size].requires_grad_(True)
        f_batch = f[i * batch_size:(i + 1) * batch_size].requires_grad_(False)

        xb_batch = xb[i * batch_size_b:(i + 1) * batch_size_b].requires_grad_(True)
        yb_batch = yb[i * batch_size_b:(i + 1) * batch_size_b].requires_grad_(True)
        ub_batch = ub[i * batch_size_b:(i + 1) * batch_size_b].requires_grad_(False)

        optimizer.zero_grad()
        loss = pinn.loss(x_batch, y_batch, xb_batch, yb_batch, f_batch, ub_batch)

        running_loss_u += n * pinn.loss_u.item()
        running_loss_b += nb * pinn.loss_b.item()

        num_samples_u += x.shape[0]
        num_samples_b += xb.shape[0]

        loss.backward()
        optimizer.step()

    running_loss = running_loss_u / num_samples_u + running_loss_b / num_samples_b

    return running_loss_u / num_samples_u, running_loss_b / num_samples_b, running_loss

