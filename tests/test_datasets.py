import equinox
import jax
import pytest
from jax import numpy as jnp

from equinox_vision.datasets import loader


@pytest.mark.parametrize("jit", [True, False])
def test_loader(jit):
    dataset = {'inputs': jnp.ones(8, 10),
               'targets': jnp.ones(8)}
    _loader = equinox.filter_jit(loader) if jit else loader
    inputs, targets = _loader(dataset, jax.random.PRNGKey(0), 3)
    assert inputs.shape[0] == 3

    inputs, targets = _loader(dataset, jax.random.PRNGKey(0), 3,
                              transform=lambda x, k: x + jax.random.normal(k, x.shape))
    assert inputs.shape[0] == 3
