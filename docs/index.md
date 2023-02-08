# equinox_vision

`equinox_vision` is a library for CIFAR-10-size image classification problems in JAX with equinox.

## Usage

```python
import equinox
import jax.random

from equinox_vision.datasets import cifar10
from equinox_vision.models import resnet
from equinox_vision.transforms import random_hflip

dataset, loader = cifar10('~/.equinox_vision/cifar10', is_train=True)
model = resnet.resnet20(key=jax.random.PRNGKey(0), num_classes=10)
model = model.train()  # or model.eval()


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
    inputs, labels = loader(dataset, batch_size=32, transform=random_hflip, key=key0)
    loss, model, opt_state = step(model, inputs, labels, opt_state)

```