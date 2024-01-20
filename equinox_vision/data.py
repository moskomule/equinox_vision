import dataclasses
from collections.abc import Callable
from pathlib import Path

import jax.random
from jax import numpy as jnp
from jaxtyping import Array
from torchvision import datasets as torch_datasets, transforms as torch_transforms


@dataclasses.dataclass(frozen=True)
class Dataset:
    inputs: Array
    labels: Array
    name: str
    size: int
    num_classes: int
    image_size: int
    num_channels: int = 3
    data_stats: tuple[tuple[float, ...], tuple[float, ...]] = None

    def __str__(self):
        return f"Dataset(name={self.name}, size={self.size}, num_classes={self.num_classes} " \
               f"image_size={self.image_size} num_channels={self.num_channels})"

    def __hash__(self):
        return hash(f"{self.inputs}{self.labels}{self.name}{self.size}{self.num_classes}"
                    f"{self.image_size}{self.num_channels}-{self.data_stats}")


def split_train_val(key: Array,
                    dataset: Dataset,
                    val_size: int
                    ) -> tuple[Dataset, Dataset]:
    indices = jax.random.permutation(key, dataset.size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    train_set = dataclasses.replace(dataset,
                                    inputs=dataset.inputs[train_indices],
                                    labels=dataset.labels[train_indices],
                                    size=len(train_indices))
    val_set = dataclasses.replace(dataset,
                                  inputs=dataset.inputs[val_indices],
                                  labels=dataset.labels[val_indices],
                                  size=len(val_indices))

    return train_set, val_set


def loader(dataset: Dataset,
           key: jax.random.PRNGKeyArray,
           batch_size: int | None = None,
           indices: Array | None = None,
           transform: Callable[[Array, jax.random.PRNGKeyArray], Array] | None = None,
           ) -> tuple[Array, Array]:
    """ Data loader function.
    If `batch_size` is specified, a batch is randomly sampled.
    If `indices` is given, a batch corresponding to `indices` is returned.
    `transform` is applied to images (dataset.inputs).
    """

    size = dataset.inputs.shape[0]
    if batch_size is not None:
        key, key1 = jax.random.split(key)
        indices = jax.random.choice(key1, size, shape=(batch_size,), replace=False)
    if indices is None:
        raise TypeError('...')
    inputs = dataset.inputs[indices]
    labels = dataset.labels[indices]

    if transform is not None:
        inputs = jax.vmap(transform)(inputs, jax.random.split(key, inputs.shape[0]))

    return inputs, labels


# mnist-like datasets
def mnist(root: str | Path,
          is_train: bool,
          download: bool = False
          ) -> Dataset:
    _dataset = torch_datasets.MNIST(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'mnist', len(inputs), 10, 28, 1)


def kmnist(root: str | Path,
           is_train: bool,
           download: bool = False
           ) -> Dataset:
    _dataset = torch_datasets.KMNIST(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'kmnist', len(inputs), 10, 28, 1)


def fmnist(root: str | Path,
           is_train: bool,
           download: bool = False
           ) -> Dataset:
    _dataset = torch_datasets.FashionMNIST(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'fmnist', len(inputs), 10, 28, 1)


# cifar-like datasets
def cifar10(root: str | Path,
            is_train: bool,
            download: bool = False
            ) -> Dataset:
    _dataset = torch_datasets.CIFAR10(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar10', len(inputs), 10, 32, 3,
                   ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))


def cifar100(root: str | Path,
             is_train: bool,
             download: bool = False
             ) -> Dataset:
    _dataset = torch_datasets.CIFAR100(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar100', len(inputs), 100, 32, 3,
                   ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))


# svhn
def svhn(root: str | Path,
         is_train: bool,
         download: bool = False
         ) -> Dataset:
    _dataset = torch_datasets.SVHN(root, 'train' if is_train else 'test',
                                   transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'svhn', len(inputs), 10, 32, 3,
                   ((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049)))
