import torch


def parameters_norm(net):
    with torch.no_grad():
        out = None
        for p in net.parameters():
            if out is None:
                out = torch.sum(p.data ** 2)
            else:
                out += torch.sum(p.data ** 2)

    return torch.sqrt(out).item()


def grad_norm(net: torch.nn.Module):

    out = torch.zeros(1)
    for p in net.parameters():
        if p.grad is None:
            continue

        out += torch.sum(p.grad.data ** 2)


def optimizer_params(optimizer):
    keys = list(optimizer.param_groups[0].keys())
    keys = tuple(keys[1:])

    names = ()
    val = []
    frmt = ''
    for name in keys:
        param = optimizer.param_groups[0][name]
        if isinstance(param, str) or isinstance(param, bool):
            continue
        elif isinstance(param, tuple):
            names += len(param) * (name,)
            val += [*param]
            for p in param:
                frmt += parameter_format(p)

        else:
            names += (name,)
            val += [param]
            frmt += parameter_format(param)

    opt_params = {'str': names, 'frmt': frmt, 'val': val}

    return opt_params


def parameter_format(param, spacing=15, decimal=4):

    if isinstance(param, int):
        frmt = '{:<' + str(spacing) + 'd}'
    else:
        frmt = '{:<' + str(spacing) + '.' + str(decimal) + 'e}'

    return frmt

