import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def load_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(dataset, num_clients, partitioning="iid", seed=42):
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    rng = np.random.default_rng(seed)
    if partitioning == "iid":
        return [Subset(dataset, chunk.tolist()) for chunk in np.array_split(rng.permutation(len(dataset)), num_clients)]
    if partitioning != "label_sorted":
        raise ValueError(f"unsupported partitioning strategy: {partitioning}")
    targets = np.asarray(dataset.targets)
    shards = np.array_split(np.argsort(targets), num_clients * 2)
    shard_order = rng.permutation(len(shards))
    return [Subset(dataset, np.concatenate([shards[i] for i in group]).tolist())
            for group in np.array_split(shard_order, num_clients)]
