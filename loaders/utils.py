import numpy as np
import os
import os.path
from PIL import Image
from loaders.randaugment import RandAugmentMC
from torchvision import transforms
import pickle
import torch.utils.data as TorchData
from typing import Callable

import torch
import torch.utils.data
import torchvision


class TransformFixMatch(object):
    def __init__(self, crop_size, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchTwo(object):
    def __init__(self, crop_size, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong2 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong2)


class ConcatDataset(TorchData.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class IterLoader:
    def __init__(self, dataloader_list):
        self.dataloader_list = dataloader_list
        self.iter_list = [iter(loader) for loader in dataloader_list]
        self.n = 0

    def __next__(self):
        self.n += 1
        return_list = []
        new_iter_list = []
        for index, iter_loader in enumerate(self.iter_list):
            data = iter_loader.next()
            return_list.append(data)
            if self.n % len(self.dataloader_list[index]) == 0:
                iter_loader = iter(self.dataloader_list[index])
            new_iter_list.append(iter_loader)
        self.iter_list = new_iter_list
        return return_list


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.labels
            # return dataset.get_labels()
        else:
            return dataset.labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
