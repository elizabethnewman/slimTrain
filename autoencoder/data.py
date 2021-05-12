
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def mnist(train_kwargs, val_kwargs, num_train=2**10, num_val=2**5, dirname=''):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3013,))
    ])

    subset_indices = torch.arange(num_train + num_val)
    dataset = torchvision.datasets.MNIST(dirname + 'mnist_data/', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs,
                                               sampler=SubsetRandomSampler(subset_indices[:num_train]))
    val_loader = torch.utils.data.DataLoader(dataset, **val_kwargs,
                                             sampler=SubsetRandomSampler(subset_indices[num_train:]))

    testset = torchvision.datasets.MNIST(dirname + 'mnist_data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, **val_kwargs)

    return train_loader, val_loader, test_loader
