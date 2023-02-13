import equinox
import jax
import pytest
from jax import numpy as jnp

from equinox_vision.data import Dataset, loader


@pytest.mark.parametrize("jit", [True, False])
def test_loader(jit):
    dataset = Dataset(jnp.ones((8, 10)), jnp.ones((8,)), 'test', 8, 2, 0, 0)
    _loader = equinox.filter_jit(loader) if jit else loader
    inputs, targets = _loader(dataset, jax.random.PRNGKey(0), 3)
    assert inputs.shape[0] == 3

    inputs, targets = _loader(dataset, jax.random.PRNGKey(0), 3,
                              transform=lambda x, k: x + jax.random.normal(k, x.shape))
    assert inputs.shape[0] == 3
