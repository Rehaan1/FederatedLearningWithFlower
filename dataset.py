import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader



def get_mnist(data_path: str="./data"):

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset



def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):

    trainset, testset = get_mnist()

    # We are considering an IID Procedure of Partitioning Data
    # Real Life Examples would be Non-IID, i.e. non-uniform

    # split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    # We will have one dataloader per client for training and validation
    trainLoaders = []
    valLoaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

        trainLoaders.append(DataLoader(for_train, batch_size=batch_size,shuffle=True, num_workers=2))
        valLoaders.append(DataLoader(for_val, batch_size=batch_size,shuffle=False, num_workers=2))

    testLoader = DataLoader(testset, batch_size=120, shuffle=True, num_workers=2)

    return trainLoaders, valLoaders, testLoader




