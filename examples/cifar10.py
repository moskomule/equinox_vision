import equinox
import jax
import optax
from rich import print
from rich.progress import track

from equinox_vision import datasets, models, transforms


def main():
    batch_size = 128
    key = jax.random.PRNGKey(0)

    key, key0 = jax.random.split(key)
    print('--setup model--')
    model = models.resnet20(key0)
    print('--setup dataset--')
    trainset = datasets.cifar10("~/.cache/equinox_vision", True, True)
    transform = jax.jit(transforms.compose([transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            transforms.random_hflip(),
                                            transforms.random_crop(32, 4, 'reflect')
                                            ]))
    # todo: how to use weight decay with optax?
    optim = optax.adamw(1e-3, weight_decay=1e-4)
    opt_state = optim.init(equinox.filter(model, equinox.is_array))

    @equinox.filter_value_and_grad
    def forward(model, inputs, labels):
        outputs = jax.vmap(model, axis_name='batch')(inputs)
        return jax.numpy.mean(optax.softmax_cross_entropy_with_integer_labels(outputs, labels))

    @equinox.filter_jit
    def train_step(model, key, opt_state):
        inputs, labels = datasets.loader(trainset, key=key, batch_size=batch_size, transform=transform)
        loss, grads = forward(model, inputs, labels)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = equinox.apply_updates(model, updates)
        return loss, model, opt_state

    print('--training--')
    for i in track(range(trainset.size // batch_size * 50)):
        key, key0 = jax.random.split(key)
        loss, model, opt_state = train_step(model, key0, opt_state)

    print('--evaluation--')
    model = model.eval()
    testset, _ = datasets.cifar10("~/.cache/equinox_vision", False, True)
    inputs = transforms.normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(testset['inputs'], None)
    preds = jax.numpy.argmax(jax.vmap(model, axis_name='batch')(inputs), axis=1)
    print(f"test accuracy {sum(preds == testset['labels']) / testset['size']}")


if __name__ == '__main__':
    main()
