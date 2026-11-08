import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(dataset, num_clients, partitioning='iid', seed=42):
    np.random.seed(seed)
    num_items = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    client_datasets = []
    
    if partitioning == 'iid':
        for i in range(num_clients):
            client_indices = indices[i * num_items : (i + 1) * num_items]
            client_datasets.append(Subset(dataset, client_indices))
    else:
        # Non-IID simplified partitioning by sorted labels
        targets = np.array(dataset.targets)
        sorted_indices = np.argsort(targets)
        shard_size = len(dataset) // (num_clients * 2)
        for i in range(num_clients):
            client_indices = sorted_indices[i * shard_size : (i + num_clients) * shard_size]
            client_datasets.append(Subset(dataset, client_indices))
            
    return client_datasets
