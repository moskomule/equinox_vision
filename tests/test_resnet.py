import jax
import pytest
from jax import numpy as jnp

from equinox_vision.models.resnet import resnet20, wrn16_8


@pytest.mark.parametrize("model", [resnet20, wrn16_8])
def test_resnet(model):
    model = model(jax.random.PRNGKey(0), num_classes=5).eval()
    model = model.train()

    # test batch norm
    data = jnp.ones((4, 3, 32, 32))
    out = jax.vmap(model, axis_name='batch')(data)
    assert out.shape[-1] == 5
