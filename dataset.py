"""
dataset.py — MNIST dataset loading, IID and non-IID partitioning.

Provides functions to download MNIST, split it into per-client datasets
(IID or non-IID), and return DataLoaders ready for local training plus a
global test DataLoader for evaluation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


def get_mnist(data_dir: str = "./data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Download and return the MNIST train and test datasets with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def partition_iid(
    dataset: datasets.MNIST,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Randomly shuffle all indices and split evenly across clients (IID).

    Returns a list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return [indices[i::num_clients].tolist() for i in range(num_clients)]


def partition_non_iid(
    dataset: datasets.MNIST,
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42,
) -> List[List[int]]:
    """
    Sort data by label and create shards; each client receives a fixed
    number of shards (default 2), so each client sees only a subset of
    classes (non-IID).

    Returns a list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)

    # Sort indices by label
    sorted_indices = np.argsort(targets)

    # Create shards
    total_shards = num_clients * shards_per_client
    shard_size = len(dataset) // total_shards
    shards = [
        sorted_indices[i * shard_size: (i + 1) * shard_size].tolist()
        for i in range(total_shards)
    ]

    # Randomly assign shards to clients
    shard_ids = np.arange(total_shards)
    rng.shuffle(shard_ids)
    client_indices = []
    for c in range(num_clients):
        assigned = shard_ids[c * shards_per_client: (c + 1) * shards_per_client]
        idx = []
        for s in assigned:
            idx.extend(shards[s])
        client_indices.append(idx)

    return client_indices


def prepare_dataloaders(
    num_clients: int,
    batch_size: int = 64,
    iid: bool = True,
    data_dir: str = "./data",
    seed: int = 42,
    use_gpu: bool = False,
    num_workers: int = 4,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Prepare per-client train DataLoaders and a global test DataLoader.

    Parameters
    ----------
    num_clients : int
        Number of FL clients.
    batch_size : int
        Batch size for training loaders.
    iid : bool
        If True, partition data IID; otherwise non-IID (shard-based).
    data_dir : str
        Directory to download / cache MNIST.
    seed : int
        Random seed for reproducibility.
    use_gpu : bool
        If True, enables pin_memory for faster CPU→GPU transfers.
    num_workers : int
        Number of DataLoader worker processes for parallel data loading.
        Use 0 on Windows if you hit multiprocessing errors.

    Returns
    -------
    train_loaders : list of DataLoader
        One DataLoader per client.
    test_loader : DataLoader
        Single global test DataLoader.
    """
    import os
    train_dataset, test_dataset = get_mnist(data_dir)

    # Partition
    if iid:
        client_indices = partition_iid(train_dataset, num_clients, seed)
    else:
        client_indices = partition_non_iid(train_dataset, num_clients, seed=seed)

    # pin_memory speeds up CPU→GPU transfer when using a GPU
    pin = use_gpu
    # Windows + CUDA can have issues with num_workers > 0 in some setups
    workers = num_workers if (use_gpu and os.name != "nt") else 0

    # Build per-client DataLoaders
    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin,
            num_workers=workers,
            persistent_workers=(workers > 0),
        )
        train_loaders.append(loader)

    # Global test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,          # larger batch is fine for evaluation
        shuffle=False,
        pin_memory=pin,
        num_workers=workers,
        persistent_workers=(workers > 0),
    )

    return train_loaders, test_loader
