import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec

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

    def __str__(self):
        return f"Dataset(name={self.name}, size={self.size}, num_classes={self.num_classes})"


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


_P = ParamSpec('_P')


def cifar10(root: str | Path,
            is_train: bool,
            download: bool = False
            ):
    _dataset = torch_datasets.cifar.CIFAR10(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar10', len(inputs), 10)


def cifar100(root: str | Path,
             is_train: bool,
             download: bool = False
             ):
    _dataset = torch_datasets.cifar.CIFAR100(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return Dataset(inputs, labels, 'cifar100', len(inputs), 100)
