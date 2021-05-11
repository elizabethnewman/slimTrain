import torch
import time
from utils import optimizer_params, parameters_norm
from networks.slimtik import SlimTikNetworkLinearOperator


def train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader,
              num_epochs=5, device='cpu', verbose=True, log_interval=1):

    opt_params = optimizer_params(optimizer)
    net_params = {'str': (), 'frmt': '', 'val': []}
    if isinstance(net, SlimTikNetworkLinearOperator):
        net_params = net.print_outs()

    results = {
        'str': ('epoch',) + opt_params['str'] + ( 'time', '|params|') + net_params['str']
               + ('running_loss', 'train_loss', 'val_loss'),
        'frmt': '{:<15d}' + opt_params['frmt'] + '{:<15.2f}{:<15.4e}' + net_params['frmt']
                + '{:<15.4e}{:<15.4e}{:<15.4e}',
        'val': None}
    results['val'] = torch.empty(0, len(results['str']))

    if verbose:
        print(('{:<15s}' * len(results['str'])).format(*results['str']))

    # initial evaluation
    train_eval = evaluate(net, criterion, train_loader, device=device)
    test_eval = evaluate(net, criterion, val_loader, device=device)

    his = len(results['str']) * [0]
    his[0] = -1
    his[-2:] = [train_eval, test_eval]
    his[results['str'].index('|params|')] = parameters_norm(net)
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(results['frmt'].format(*his))

    total_start = time.time()

    for epoch in range(num_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader, device=device)
        end = time.time()

        train_eval = evaluate(net, criterion, train_loader, device=device)
        test_eval = evaluate(net, criterion, val_loader, device=device)

        # norm of network weights
        param_nrm = parameters_norm(net)
        opt_params = optimizer_params(optimizer)
        net_params = {'str': (), 'frmt': '', 'val': []}
        if isinstance(net, SlimTikNetworkLinearOperator):
            net_params = net.print_outs()

        # store results
        his = [epoch]
        his += opt_params['val']
        his += [end - start, param_nrm]
        his += net_params['val']
        his +=[train_out]
        his += [train_eval, test_eval]
        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose and not epoch % log_interval:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results, total_end - total_start


def train_one_epoch(model, criterion, optimizer, train_loader, device='cpu'):
    model.train()
    running_loss = 0
    num_samples = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        num_samples += inputs.shape[0]

        optimizer.zero_grad()
        if isinstance(model, SlimTikNetworkLinearOperator):
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
    num_samples = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            num_samples += inputs.shape[0]
            output = model(inputs)

            test_loss += criterion(output, inputs.to(output.device)).item()

    return test_loss / num_samples


