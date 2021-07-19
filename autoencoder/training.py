import torch
import time
from utils import optimizer_params, extract_parameters
from autoencoder.mnist import MNISTAutoencoderSlimTik


def train_sgd(net, criterion, optimizer, scheduler, train_loader, val_loader,
              num_epochs=5, device='cpu', verbose=True, log_interval=1,  save_intermediate=False):

    # ---------------------------------------------------------------------------------------------------------------- #
    # setup printouts
    # ---------------------------------------------------------------------------------------------------------------- #

    # optimization parameters
    opt_params = optimizer_params(optimizer)

    # network parameters
    net_params = {'str': (), 'frmt': '', 'val': []}
    if isinstance(net, MNISTAutoencoderSlimTik):
        net_params = net.print_outs()

    # store results and printout headings/formats
    results = {
        'str': ('epoch',) + opt_params['str'] + ('time', '|th|', '|th-th_old|', '|grad_th|') + net_params['str']
               + ('running_loss', 'train_loss', 'val_loss'),
        'frmt': '{:<15d}' + opt_params['frmt'] + '{:<15.2f}{:<15.4e}{:<15.4e}{:<15.4e}' + net_params['frmt']
                + '{:<15.4e}{:<15.4e}{:<15.4e}',
        'val': None}
    results['val'] = torch.empty(0, len(results['str']))

    # print headings
    if verbose:
        print(('{:<15s}' * len(results['str'])).format(*results['str']))

    # ---------------------------------------------------------------------------------------------------------------- #
    # initial network evaluation
    # ---------------------------------------------------------------------------------------------------------------- #

    # evaluate training and testing/validation data
    train_eval = evaluate(net, criterion, train_loader, device=device)
    test_eval = evaluate(net, criterion, val_loader, device=device)
    intermediate_results = []
    if save_intermediate:
        intermediate_results.append(store_intermediate_approx(net, train_loader, device=device))

    # optimal validation loss
    opt_val_loss = test_eval
    opt_val_loss_net = net

    # network weights
    old_params = extract_parameters(net).clone()

    his = len(results['str']) * [0]
    his[0] = -1
    his[-2:] = [train_eval, test_eval]
    his[results['str'].index('|th|')] = torch.norm(old_params).item()
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(results['frmt'].format(*his))

    # ---------------------------------------------------------------------------------------------------------------- #
    # main iteration
    # ---------------------------------------------------------------------------------------------------------------- #

    # start total training time
    total_start = time.time()

    for epoch in range(num_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader, device=device)
        end = time.time()

        train_eval = evaluate(net, criterion, train_loader, device=device)
        test_eval = evaluate(net, criterion, val_loader, device=device)

        if save_intermediate:
            intermediate_results.append(store_intermediate_approx(net, train_loader, device=device))

        if test_eval < opt_val_loss:
            opt_val_loss = test_eval
            opt_val_loss_net = net

        # optimization parameters
        opt_params = optimizer_params(optimizer)

        # network parameters
        if isinstance(net, MNISTAutoencoderSlimTik):
            net_params = net.print_outs()

        # network weights
        new_params = extract_parameters(net)

        # norm of network weights
        param_nrm = torch.norm(new_params).item()

        # store results
        his = [epoch]
        his += opt_params['val']
        his += [end - start, param_nrm, torch.norm(new_params - old_params).item(), 0]
        his += net_params['val']
        his += [train_out]
        his += [train_eval, test_eval]
        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose and not epoch % log_interval:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

        # store past network weights
        old_params = new_params.clone()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results, total_end - total_start, opt_val_loss_net, intermediate_results


def train_one_epoch(net, criterion, optimizer, train_loader, device='cpu'):
    net.train()
    running_loss = 0
    num_samples = 0
    criterion.reduction = 'mean'

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        num_samples += inputs.shape[0]

        optimizer.zero_grad()

        if isinstance(net, MNISTAutoencoderSlimTik):
            output = net(inputs, inputs)
        else:
            output = net(inputs)

        loss = criterion(output, inputs.to(output.device))
        running_loss += inputs.shape[0] * loss.item()

        # # average over the batch; criterion must have reduction='sum'
        # loss = loss / inputs.shape[0]

        loss.backward()
        optimizer.step()

    return running_loss / num_samples


def evaluate(net, criterion, data_loader, device='cpu'):
    net.eval()
    test_loss = 0
    num_samples = 0
    criterion.reduction = 'sum'
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            num_samples += inputs.shape[0]
            output = net(inputs)
            test_loss += criterion(output, inputs.to(output.device)).item()

    return test_loss / num_samples


def store_intermediate_approx(net, data_loader, device='cpu'):
    net.eval()
    with torch.no_grad():
        inputs = data_loader.dataset.data[:32].unsqueeze(1)
        labels = data_loader.dataset.targets[:32]
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        loss = torch.sum((output - inputs.to(output.device)) ** 2, dim=(1, 2, 3))

    return {'output': output.to('cpu'), 'loss': loss.to('cpu')}


