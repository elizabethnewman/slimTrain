import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from mpl_toolkits.mplot3d import Axes3D

import torch
import numpy as np


# main peaks function
def peaks_function(y: torch.Tensor) -> torch.Tensor:
    F = (3 * (1 - y[:, 0])**2 * torch.exp(-1.0 * (y[:, 0]**2 + (y[:, 1] + 1)**2))
         - 10 * (y[:, 0] / 5 - y[:, 0]**3 - y[:, 1]**5) * torch.exp(-y[:, 0]**2 - y[:, 1]**2)
         - (1 / 3) * torch.exp(-(y[:, 0]+1)**2 - y[:, 1]**2))
    return F


# helper functions
def get_grid2D(domain_x=(-3, 3), domain_y=(-3, 3), step=0.25):
    x = torch.arange(domain_x[0], domain_x[1] + step, step)
    y = torch.arange(domain_y[0], domain_y[1] + step, step)
    grid_x, grid_y = torch.meshgrid(x, y)

    return grid_x, grid_y


def get_regression_data(n, domain=(-3, 3)):
    
    y = (domain[1] - domain[0]) * (torch.rand(2 * n, 2, requires_grad=False) - 0.5)
    c = peaks_function(y).view(-1, 1)

    return y, c


def visualize_regression_image(net, fig_num=1):
    grid_x, grid_y = get_grid2D()

    grid_data = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)

    # true
    c_true = peaks_function(grid_data).detach().view(grid_x.shape).numpy()

    # approximation
    c_pred = net.forward(grid_data).detach().view(grid_x.shape).numpy()

    print('relative error = ',
          np.linalg.norm(c_pred.reshape(-1) - c_true.reshape(-1)) / np.linalg.norm(c_true.reshape(-1)))

    # image plots
    plt.figure(fig_num)
    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    p = ax.imshow(c_true.reshape(grid_x.shape))
    ax.axis('off')
    ax.set_title('true')
    fig.colorbar(p, ax=ax, aspect=10)

    ax = axs[0, 1]
    p = ax.imshow(c_pred.reshape(grid_x.shape))
    ax.axis('off')
    ax.set_title('approx')
    fig.colorbar(p, ax=ax, aspect=10)

    ax = axs[1, 0]
    p = ax.imshow(np.abs(c_pred - c_true).reshape(grid_x.shape))
    fig.colorbar(p, ax=ax, aspect=10)
    ax.axis('off')
    ax.set_title('abs. diff.')

    ax = axs[1, 1]
    ax.axis('off')


def visualize_regression_plot3D(net):
    grid_x, grid_y = get_grid2D()

    grid_data = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), dim=1)

    # true
    c_true = peaks_function(grid_data).detach().view(grid_x.shape).numpy()

    # approximation
    c_pred = net.forward(grid_data).detach().view(grid_x.shape).numpy()

    print('relative error = ',
          np.linalg.norm(c_pred.reshape(-1) - c_true.reshape(-1)) / np.linalg.norm(c_true.reshape(-1)))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, c_true, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('true')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, c_pred, cmap=cm.viridis, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('approximation')

    # fig = plt.figure()
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, np.abs(c_pred - c_true), cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('abs. difference')