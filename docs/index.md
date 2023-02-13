# equinox_vision

`equinox_vision` is a library for CIFAR-10-size image classification problems in JAX with equinox.

## Usage

```python
import equinox
import jax.random

from equinox_vision.data import cifar10, loader
from equinox_vision.models import resnet
from equinox_vision.transforms import compose, normalize, random_hflip

dataset = cifar10('~/.equinox_vision/cifar10', is_train=True)
model = resnet.resnet20(key=jax.random.PRNGKey(0), num_classes=10)
model = model.train()  # or model.eval()

transform = compose([
    normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    random_hflip()])  # (1)


@equinox.filter_value_and_grad
def forward(model, inputs, labels):
    outputs = jax.vmap(model, axis_name='batch')(inputs)
    return loss(outputs, labels)


@equinox.filter_jit
def step(model, key, opt_state):
    inputs, labels = loader(dataset, key=key,
                            batch_size=32, transform=transform)  # (2)
    loss, grads = forward(model, inputs, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = equinox.apply_updates(model, updates)
    return loss, model, opt_state


for i in range(num_iters):
    key, key0 = jax.random.split(key)
    loss, model, opt_state = step(model, key0, opt_state)

```

1. Multiple transformations for data augmentation can be composed with [equinox_vision.transforms.compose][].
2. Datasets can be loaded with jit'able [equinox_vision.data.loader][].

## APIs

### Models

#### ResNets

::: equinox_vision.models.resnet20
::: equinox_vision.models.resnet56
::: equinox_vision.models.resnet20_gn
::: equinox_vision.models.resnet56_gn

#### WideResNets

::: equinox_vision.models.wrn28_2
::: equinox_vision.models.wrn40_2
::: equinox_vision.models.wrn28_2_gn
::: equinox_vision.models.wrn40_2_gn

### Datasets

::: equinox_vision.data.mnist
::: equinox_vision.data.fmnist
::: equinox_vision.data.kmnist
::: equinox_vision.data.cifar10
::: equinox_vision.data.cifar100
::: equinox_vision.data.svhn

### Data loader

::: equinox_vision.data.loader

### Transforms

Image transformation functions for data augmentation.

::: equinox_vision.transforms.random_hflip
::: equinox_vision.transforms.random_crop
::: equinox_vision.transforms.normalize
::: equinox_vision.transforms.compose
