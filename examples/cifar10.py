import enum

import equinox
import jax
import optax
from jax import numpy as jnp
from rich import print
from rich.progress import track

from equinox_vision import data, models, transforms


class Model(enum.StrEnum):
    wrn28_2 = enum.auto()
    wrn28_2_gn = enum.auto()
    resnet56 = enum.auto()
    resnet56_gn = enum.auto()

    @property
    def create(self):
        return getattr(models, self.value)


def main(model: Model):
    batch_size = 128
    num_iters = 80_000

    key = jax.random.PRNGKey(0)
    key, key0 = jax.random.split(key)

    print('--setup model--')
    model = Model(model).create(key, return_state=True)
    state = equinox.nn.State(model)

    print('--setup dataset--')
    trainset = data.cifar10("~/.cache/equinox_vision", True, True)
    testset = data.cifar10("~/.cache/equinox_vision", False, True)
    transform = transforms.compose([transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    transforms.random_crop(32, 4, 'reflect'),
                                    transforms.random_hflip()])
    optim = optax.chain(optax.add_decayed_weights(1e-4),
                        optax.sgd(optax.warmup_cosine_decay_schedule(0.01, 0.1,
                                                                     warmup_steps=int(0.01 * num_iters),
                                                                     decay_steps=int(0.99 * num_iters)),
                                  momentum=0.9))
    opt_state = optim.init(equinox.filter(model, equinox.is_inexact_array))

    def forward(model, state, inputs, labels):
        outputs, state = jax.vmap(model, axis_name='batch', in_axes=(0, None), out_axes=(0, None))(inputs, state)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(outputs, labels)), state

    @equinox.filter_jit
    def train_step(model, state, key, opt_state):
        inputs, labels = data.loader(trainset, key=key, batch_size=batch_size, transform=transform)
        (loss, state), grads = equinox.filter_value_and_grad(forward, has_aux=True)(model, state, inputs, labels)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = equinox.apply_updates(model, updates)
        return loss, state, model, opt_state

    @equinox.filter_jit
    def val_step(model, inputs, labels):
        inputs = transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(inputs)
        outputs, _ = jax.vmap(model)(inputs)
        preds = jnp.argmax(outputs, axis=1)
        return jnp.sum(preds == labels) / preds.shape[0]

    print('--training--')
    for i in track(range(num_iters)):
        key, key0 = jax.random.split(key)
        loss, state, model, opt_state = train_step(model, state, key0, opt_state)
        if i % (num_iters // 100) == 0:
            print(f"{float(loss):.4f}")
            accuracy = val_step(equinox.Partial(equinox.tree_inference(model, value=True), state=state),
                                testset.inputs, testset.labels)
            print(f"test accuracy at {i:>10}th iteration: {accuracy:.3f}")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=[e.value for e in Model], default=Model.wrn28_2_gn)
    args = p.parse_args()
    main(args.model)
