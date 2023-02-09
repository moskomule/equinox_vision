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

    def __str__(self):
        return f"Dataset(name={self.name}, size={self.size}, num_classes={self.num_classes} " \
               f"image_size={self.image_size} num_channels={self.num_channels})"

    def __hash__(self):
        return hash(f"{self.inputs}{self.labels}{self.name}{self.size}{self.num_classes}"
                    f"{self.image_size}{self.num_channels}")


def loader(dataset: Dataset,
           key: jax.random.PRNGKeyArray,
           batch_size: int | None = None,
           indices: Array | None = None,
           transform: Callable[[Array, jax.random.PRNGKeyArray], Array] | None = None,
           ) -> tuple[Array, Array]:
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


fashion_mnist = fmnist


# cifar-like datasets
def cifar10(root: str | Path,
            is_train: bool,
            download: bool = False
            ) -> Dataset:
    _dataset = torch_datasets.CIFAR10(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar10', len(inputs), 10, 32)


def cifar100(root: str | Path,
             is_train: bool,
             download: bool = False
             ) -> Dataset:
    _dataset = torch_datasets.CIFAR100(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar100', len(inputs), 100, 32)
