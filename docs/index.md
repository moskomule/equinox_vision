# equinox_vision

`equinox_vision` is a library for CIFAR-10-size image classification problems in JAX with equinox.

## Usage

```python
import equinox
import jax.random

from equinox_vision.datasets import cifar10
from equinox_vision.models import resnet
from equinox_vision.transforms import compose, normalize, random_hflip

dataset, loader = cifar10('~/.equinox_vision/cifar10', is_train=True)  # (1)
model = resnet.resnet20(key=jax.random.PRNGKey(0), num_classes=10)
model = model.train()  # or model.eval()

transform = jax.jit(compose([
    normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    random_hflip()])
)  # (2)


@equinox.filter_value_and_grad
def forward(model, inputs, labels):
    outputs = jax.vmap(model, axis_name='batch')(inputs)
    return loss(outputs, labels)


@equinox.filter_jit
def step(model, inputs, labels, opt_state):
    loss, grads = forward(model, inputs, labels)
    updates, opt_state = optim.update(grads, opt_state)
    model = equinox.apply_updates(model, updates)
    return loss, model, opt_state


for i in range(num_iters):
    key, key0 = jax.random.split(key)
    inputs, labels = loader(dataset, batch_size=32, transform=transform, key=key0)
    loss, model, opt_state = step(model, inputs, labels, opt_state)

```

1. Each dataset creation function consists of `dataset` (dict) and `loader` (function).
2. Multiple transformations can be composed.