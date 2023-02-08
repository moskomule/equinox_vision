import equinox
import jax
import pytest
from jax import numpy as jnp

from equinox_vision.resnet import resnet20, wrn16_8


@pytest.mark.parametrize("model", [resnet20, wrn16_8])
def test_resnet(model):
    model = model(num_classes=5)

    data = jnp.ones(3, 32, 32)

    assert model(data).shape[0] == 5

    # test batch norm
    data = jnp.ones(4, 3, 32, 32)
    jax.vmap(equinox.filter(model, equinox.is_array), axis_name='batch')(data)
