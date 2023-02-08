import functools
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec

import jax.random
from jax import numpy as jnp
from jaxtyping import Array
from torchvision import datasets as torch_datasets, transforms as torch_transforms

_P = ParamSpec('_P')


@jax.jit
def loader(dataset: dict[str, Array],
           key: jax.random.PRNGKeyArray,
           batch_size: int | None = None,
           indices: Array | None = None,
           transform: Callable[[Array, jax.random.PRNGKeyArray], Array] | None = None,
           ) -> tuple[Array, Array]:
    size = dataset['inputs'].shape[0]
    if batch_size is not None:
        key, key1 = jax.random.split(key)
        indices = jax.random.permutation(key1, size)
    if indices is None:
        raise TypeError('...')
    inputs = dataset['inputs'][indices]
    labels = dataset['labels'][indices]

    if transform is not None:
        inputs = jax.vmap(transform)(inputs, jax.random.split(key, size))

    return inputs, labels


def classification_dataset(f: Callable[_P, dict[str, Array]]) -> Callable[_P, dict[str, Array], Callable]:
    @functools.wraps(f)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> tuple[dict[str, Array], Callable]:
        dataset = f(*args, **kwargs)
        assert set(dataset.keys()) == {'inputs', 'labels'}
        return dataset, loader

    return wrapped


@classification_dataset
def cifar10(root: str | Path,
            is_train: bool,
            download: bool = False
            ):
    _dataset = torch_datasets.cifar.CIFAR10(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return {"inputs": inputs, "labels": labels}


@classification_dataset
def cifar100(root: str | Path,
             is_train: bool,
             download: bool = False
             ):
    _dataset = torch_datasets.cifar.CIFAR100(root, is_train, transform=torch_transforms.ToTensor(), download=download)
    inputs = jnp.stack([img.numpy() for img, label in _dataset])  # BCHW in [0, 1]
    labels = jnp.array([label for img, label in _dataset])
    return {"inputs": inputs, "labels": labels}
