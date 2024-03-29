# inspired by deepmind/dm_pix
import functools
from collections.abc import Callable
from typing import TypeAlias

import jax
from jax import numpy as jnp
from jaxtyping import Array

TransformFunc: TypeAlias = Callable[[Array, Array], Array]


def random_hflip(prob: float = 0.5) -> TransformFunc:
    """ Random horizontal flipping.
    """

    def _random_hflip(img: Array,
                      key: Array
                      ) -> Array:
        coin = jax.random.bernoulli(key, prob)
        return jax.lax.cond(coin,
                            lambda x: jnp.flip(x, -1),  # true_fn
                            lambda x: x,  # false_fn
                            img)

    return _random_hflip


def random_crop(size: int,
                padding: int = 0,
                padding_mode: str = 'constant') -> TransformFunc:
    """ Random cropping with optional padding (`padding`, `padding_mode`) for squared images.
    See [jax.numpy.padding]()'s `mode` for `padding_mode` options.
    """

    slice_sizes = (size, size)

    def _random_crop(img: Array,
                     key: Array
                     ) -> Array:
        img_size = img.shape[-1]
        if padding > 0:
            pad = functools.partial(jnp.pad, pad_width=padding, mode=padding_mode)
            img = jax.vmap(pad)(img) if img.ndim == 3 else pad(img)
        start_indices = [jax.random.randint(key, (), 0, img_size + 2 * padding - size + 1) for key in
                         jax.random.split(key)]
        crop = functools.partial(jax.lax.dynamic_slice, start_indices=start_indices, slice_sizes=slice_sizes)
        return jax.vmap(crop)(img) if img.ndim == 3 else crop(img)

    return _random_crop


def normalize(mean: tuple | Array,
              std: tuple | Array
              ) -> TransformFunc:
    """ Normalization of inputs.
    """

    def _ensure_3d(arr):
        if isinstance(arr, tuple):
            arr = jnp.array(arr)
        if arr.ndim == 1:
            return arr[:, None, None]
        else:
            return arr

    mean, std = _ensure_3d(mean), _ensure_3d(std)

    def _normalize(img: Array,
                   key: Array = None
                   ) -> Array:
        return (img - mean) / std

    return _normalize


def compose(funcs: list[TransformFunc]
            ) -> TransformFunc:
    """ Compose several transformations.
    """

    num_funcs = len(funcs)

    def composed(img: Array, key: Array) -> Array:
        keys = jax.random.split(key, num_funcs)
        for func, key in zip(funcs, keys):
            img = func(img, key)
        return img

    return composed
