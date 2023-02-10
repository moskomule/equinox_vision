import equinox
import jax
import optax
from jax import numpy as jnp
from rich import print
from rich.progress import track

from equinox_vision import datasets, models, transforms


def main():
    batch_size = 128
    num_iters = 80_000
    key = jax.random.PRNGKey(0)

    key, key0 = jax.random.split(key)
    print('--setup model--')
    model = models.wrn28_2_gn(key0).train()  # batch_norm is too slow...
    print('--setup dataset--')
    trainset = datasets.cifar10("~/.cache/equinox_vision", True, True)
    testset = datasets.cifar10("~/.cache/equinox_vision", False, True)
    transform = transforms.compose([transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    transforms.random_crop(32, 4, 'reflect'),
                                    transforms.random_hflip()])
    optim = optax.chain(optax.additive_weight_decay(1e-4),
                        optax.sgd(optax.warmup_cosine_decay_schedule(0.01, 0.1,
                                                                     warmup_steps=int(0.01 * num_iters),
                                                                     decay_steps=int(0.99 * num_iters)),
                                  momentum=0.9))
    opt_state = optim.init(equinox.filter(model, equinox.is_array))

    @equinox.filter_value_and_grad
    def forward(model, inputs, labels):
        outputs = jax.vmap(model, axis_name='batch')(inputs)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(outputs, labels))

    @equinox.filter_jit
    def train_step(model, key, opt_state):
        inputs, labels = datasets.loader(trainset, key=key, batch_size=batch_size, transform=transform)
        loss, grads = forward(model, inputs, labels)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = equinox.apply_updates(model, updates)
        return loss, model, opt_state

    @equinox.filter_jit
    def val_step(model, inputs, labels):
        inputs = transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(inputs)
        preds = jnp.argmax(jax.vmap(model, axis_name='batch')(inputs), axis=1)
        return jnp.sum(preds == labels) / preds.shape[0]

    print('--training--')
    for i in track(range(num_iters)):
        key, key0 = jax.random.split(key)
        loss, model, opt_state = train_step(model, key0, opt_state)
        if i % (num_iters // 100) == 0:
            accuracy = val_step(model.eval(), testset.inputs, testset.labels)
            print(f"test accuracy at {i:>10}th iteration: {accuracy:.3f}")


if __name__ == '__main__':
    main()
