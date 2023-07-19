import enum

import equinox
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
import optax
from jax import numpy as jnp
from rich import print
from rich.progress import track

from equinox_vision import data, models, transforms


class Model(enum.StrEnum):
    wrn_28_2 = enum.auto()
    wrn_28_2_gn = enum.auto()

    @property
    def create(self):
        match self:
            case Model.wrn_28_2:
                return models.wrn28_2
            case Model.wrn_28_2_gn:
                return models.wrn28_2_gn
            case _:
                raise NotImplementedError()


def main(distributed: bool, model: Model):
    batch_size = 128
    num_iters = 80_000

    key = jax.random.PRNGKey(0)
    key, key0 = jax.random.split(key)

    print('--setup model--')
    model = model.create(key)
    model.return_state = True  # to share code
    state = equinox.nn.State(model)

    print('--setup dataset--')
    trainset = data.cifar10("~/.cache/equinox_vision", True, True)
    testset = data.cifar10("~/.cache/equinox_vision", False, True)
    transform = transforms.compose([transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    transforms.random_crop(32, 4, 'reflect'),
                                    transforms.random_hflip()])
    optim = optax.chain(optax.additive_weight_decay(1e-4),
                        optax.sgd(optax.warmup_cosine_decay_schedule(0.01, 0.1,
                                                                     warmup_steps=int(0.01 * num_iters),
                                                                     decay_steps=int(0.99 * num_iters)),
                                  momentum=0.9))
    opt_state = optim.init(equinox.filter(model, equinox.is_array))

    num_devices = len(jax.devices())
    if num_devices > 1 and distributed:
        devices = mesh_utils.create_device_mesh((num_devices, 1))
        shard = sharding.PositionalSharding(devices)
        print(f"{num_devices} devices found; start distributed mode")

    def forward(model, state, inputs, labels):
        outputs, state = jax.vmap(model, axis_name='batch')(inputs, state)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(outputs, labels)), state

    @equinox.filter_jit
    def train_step(model, state, key, opt_state):
        inputs, labels = data.loader(trainset, key=key, batch_size=batch_size, transform=transform)
        if num_devices > 1 and distributed:
            inputs, labels = jax.device_put((inputs, labels), shard)
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
            accuracy = val_step(model.eval(), testset.inputs, testset.labels)
            print(f"test accuracy at {i:>10}th iteration: {accuracy:.3f}")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--distributed", action='store_true')
    p.add_argument("--model", choices=[Model.wrn_28_2, Model.wrn_28_2_gn], default=Model.wrn_28_2_gn)
    args = p.parse_args()
    main(args.distributed, args.model)
