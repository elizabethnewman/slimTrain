import torch
import time
from utils import optimizer_params, parameters_norm, extract_parameters, extract_gradients
from networks.slimtik import SlimTikNetworkLinearOperator
from slimtik_functions.get_convolution_matrix import get_ConcatenatedConv2DTranspose_matrix, get_Conv2DTranspose_matrix


def train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader,
              num_epochs=5, device='cpu', verbose=True, log_interval=1):

    opt_params = optimizer_params(optimizer)
    net_params = {'str': (), 'frmt': '', 'val': []}
    if isinstance(net, SlimTikNetworkLinearOperator) or isinstance(net, SlimTikNetworkLinearOperatorFull):
        net_params = net.print_outs()

    results = {
        'str': ('epoch',) + opt_params['str'] + ('time', '|th|', '|th-th_old|', '|grad_th|') + net_params['str']
               + ('running_loss', 'train_loss', 'val_loss'),
        'frmt': '{:<15d}' + opt_params['frmt'] + '{:<15.2f}{:<15.4e}{:<15.4e}{:<15.4e}' + net_params['frmt']
                + '{:<15.4e}{:<15.4e}{:<15.4e}',
        'val': None}
    results['val'] = torch.empty(0, len(results['str']))

    if verbose:
        print(('{:<15s}' * len(results['str'])).format(*results['str']))

    # initial evaluation
    train_eval, _ = evaluate(net, criterion, train_loader, device=device)
    test_eval, _ = evaluate(net, criterion, val_loader, device=device)

    old_params = extract_parameters(net).clone()

    his = len(results['str']) * [0]
    his[0] = -1
    his[-2:] = [train_eval, test_eval]
    his[results['str'].index('|th|')] = torch.norm(old_params).item()
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(results['frmt'].format(*his))

    total_start = time.time()

    for epoch in range(num_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader, device=device)
        end = time.time()

        train_eval, full_grad_nrm = evaluate(net, criterion, train_loader, device=device)
        test_eval, test_grad = evaluate(net, criterion, val_loader, device=device)

        new_params = extract_parameters(net)
        # full_grad = compute_full_gradient(net, criterion, train_loader)
        # norm of network weights
        param_nrm = torch.norm(new_params).item()
        # full_grad_nrm = torch.norm(full_grad)
        opt_params = optimizer_params(optimizer)
        net_params = {'str': (), 'frmt': '', 'val': []}
        if isinstance(net, SlimTikNetworkLinearOperator) or isinstance(net, SlimTikNetworkLinearOperatorFull):
            net_params = net.print_outs()

        # store results
        his = [epoch]
        his += opt_params['val']
        his += [end - start, param_nrm, torch.norm(new_params - old_params).item(), full_grad_nrm]
        his += net_params['val']
        his += [train_out]
        his += [train_eval, test_eval]
        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose and not epoch % log_interval:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

        old_params = new_params.clone()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results, total_end - total_start


def train_one_epoch(model, criterion, optimizer, train_loader, device='cpu'):
    model.train()
    running_loss = 0
    num_samples = 0
    criterion.reduction = 'sum'

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        num_samples += inputs.shape[0]

        optimizer.zero_grad()
        if isinstance(model, SlimTikNetworkLinearOperator) or isinstance(model, SlimTikNetworkLinearOperatorFull):
            output = model(inputs, inputs)
        else:
            output = model(inputs)

        loss = criterion(output, inputs.to(output.device))
        running_loss += loss.item()

        # average over the batch; assume criterion does not have reduction='mean'
        loss = loss / inputs.shape[0]

        loss.backward()
        optimizer.step()

    return running_loss / num_samples


def evaluate(model, criterion, data_loader, device='cpu'):
    model.eval()
    test_loss = 0
    test_grad = torch.zeros(model.W.numel())
    num_samples = 0
    criterion.reduction = 'sum'
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            num_samples += inputs.shape[0]
            output = model(inputs)

            test_loss += criterion(output, inputs.to(output.device)).item()

            Z = get_Conv2DTranspose_matrix(model.linOp)
            W_grad = Z.t() @ (Z @ model.W.reshape(-1, 1) - inputs.to(output.device).reshape(-1, 1)) \
                     + model.Lambda * model.W.reshape(-1, 1)
            test_grad += W_grad.reshape(-1) / inputs.shape[0]

    return test_loss / num_samples, torch.norm(test_grad).item()


# def compute_full_gradient(model, criterion, data_loader, device='cpu'):
#     model.train()
#     test_loss = 0
#     num_samples = 0
#     criterion.reduction = 'sum'
#     full_grad = 0
#     for i, (inputs, labels) in enumerate(data_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         num_samples += inputs.shape[0]
#         output = model(inputs)
#         loss = criterion(output, inputs.to(output.device))
#
#         # should we average?
#         loss = loss / inputs.shape[0]
#         loss.backward()
#         tmp = extract_gradients(model)
#         full_grad += tmp
#
#     return full_grad
