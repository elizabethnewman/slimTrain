import torch
import time
from utils import optimizer_params, parameters_norm
from networks.slimtik import SlimTikNetwork


def train_sgd(net, criterion, optimizer, scheduler, y_train, c_train, y_test, c_test,
          num_epochs=5, batch_size=10, verbose=True):

    opt_params = optimizer_params(optimizer)
    net_params = {'str': (), 'frmt': '', 'val': []}
    if isinstance(net, SlimTikNetwork):
        net_params = net.print_outs()

    results = {
        'str': ('epoch',) + opt_params['str'] + ( 'time', '|params|') + net_params['str']
               + ('running_loss', 'running_acc', 'loss', 'acc', 'loss', 'acc'),
        'frmt': '{:<15d}' + opt_params['frmt'] + '{:<15.2f}{:<15.4e}' + net_params['frmt']
                + '{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}',
        'val': None}
    results['val'] = torch.empty(0, len(results['str']))

    # headings
    top_header = len(results['str']) * ['']
    top_header[-2] = 'test'
    top_header[-6] = 'train'

    if verbose:
        print((len(results['str']) * '{:<15s}').format(*top_header))
        print(('{:<15s}' * len(results['str'])).format(*results['str']))

    # initial evaluation
    train_eval = evaluate(net, criterion, y_train, c_train, batch_size=batch_size)
    test_eval = evaluate(net, criterion, y_test, c_test, batch_size=batch_size)

    his = len(results['str']) * [0]
    his[0] = -1
    his[-4:] = [*train_eval, *test_eval]
    his[results['str'].index('|params|')] = parameters_norm(net)
    results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

    if verbose:
        print(results['frmt'].format(*his))

    total_start = time.time()

    for epoch in range(num_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, y_train, c_train, batch_size=batch_size)
        end = time.time()

        train_eval = evaluate(net, criterion, y_train, c_train, batch_size=batch_size)
        test_eval = evaluate(net, criterion, y_test, c_test, batch_size=batch_size)

        # norm of network weights
        param_nrm = parameters_norm(net)
        opt_params = optimizer_params(optimizer)
        net_params = {'str': (), 'frmt': '', 'val': []}
        if isinstance(net, SlimTikNetwork):
            net_params = net.print_outs()

        # store results
        his = [epoch]
        his += opt_params['val']
        his += [end - start, param_nrm]
        his += net_params['val']
        his +=[*train_out]
        his += [*train_eval, *test_eval]
        results['val'] = torch.cat((results['val'], torch.tensor(his).reshape(1, -1)), dim=0)

        if verbose:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results, total_end


def train_one_epoch(model, criterion, optimizer, train_data, train_labels, batch_size=10):
    model.train()
    running_loss = 0
    correct = 0
    num_samples = 0

    N = train_data.shape[0]
    num_batches = N // batch_size

    batch_idx = torch.randperm(N)
    train_data = train_data[batch_idx]
    train_labels = train_labels[batch_idx]
    for i in range(num_batches):

        data = train_data[i * batch_size:(i + 1) * batch_size]
        target = train_labels[i * batch_size:(i + 1) * batch_size]

        optimizer.zero_grad()

        if isinstance(model, SlimTikNetwork):
            output = model(data, target)
        else:
            output = model(data)

        loss = criterion(output, target)
        running_loss += loss.item()

        num_samples += data.shape[0]

        pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

    return running_loss / num_samples, 100 * correct / num_samples


def evaluate(model, criterion, test_data, test_labels, batch_size=10):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        N = test_data.shape[0]
        num_batches = N // batch_size

        for i in range(num_batches):
            data = test_data[i * batch_size:(i + 1) * batch_size]
            target = test_labels[i * batch_size:(i + 1) * batch_size]

            output = model(data)

            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= N

    return test_loss, 100. * correct / N
