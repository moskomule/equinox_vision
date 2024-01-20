import equinox
import jax
import pytest
from jax import numpy as jnp

from equinox_vision.models.resnet import resnet20, resnet20_gn, wrn16_8, wrn28_2_gn


@pytest.mark.parametrize("model", [resnet20, wrn16_8])
def test_resnet_bn(model):
    model, state = equinox.nn.make_with_state(model)(jax.random.PRNGKey(0), num_classes=5)
    model = model.train()

    # test batch norm
    data = jnp.ones((4, 3, 32, 32))
    out, _ = jax.vmap(model, axis_name='batch', in_axes=(0, None), out_axes=(0, None))(data)
    assert out.shape[-1] == 5


@pytest.mark.parametrize("model", [resnet20_gn, wrn28_2_gn])
def test_resnet_gn(model):
    model = model(jax.random.PRNGKey(0), num_classes=5)
    model = model.train()

    data = jnp.ones((4, 3, 32, 32))
    out = jax.vmap(model)(data)
    assert out.shape[-1] == 5
