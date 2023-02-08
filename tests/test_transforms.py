import jax
import pytest

from equinox_vision.transforms import compose, normalize, random_crop, random_hflip


@pytest.mark.parametrize("jit", [True, False])
def test_random_hflip(jit):
    img = jax.numpy.ones(3, 32, 32)
    f = random_hflip()
    _random_hflip = jax.jit(f) if jit else f
    _random_hflip(img, jax.random.PRNGKey(0))


@pytest.mark.parametrize("jit", [True, False])
def test_random_crop(jit):
    img = jax.numpy.ones(3, 32, 32)
    f = random_crop(32, 4)
    _random_crop = jax.jit(f) if jit else f
    out = _random_crop(img, jax.random.PRNGKey(0))
    assert out.shape[-1] == 32


@pytest.mark.parametrize("jit", [True, False])
def test_normalize(jit):
    img = jax.numpy.ones(3, 32, 32)
    f = normalize(jax.numpy.array((0.4914, 0.4822, 0.4465)), jax.numpy.array((0.2023, 0.1994, 0.2010)))
    _normalize = jax.jit(f) if jit else f
    _normalize(img, jax.random.PRNGKey(0))


@pytest.mark.parametrize("jit", [True, False])
def test_compose(jit):
    img = jax.numpy.ones(3, 32, 32)
    f = compose([random_hflip(), random_hflip()])
    _compose = jax.jit(f) if jit else f
    _compose(img, jax.random.PRNGKey(0))
