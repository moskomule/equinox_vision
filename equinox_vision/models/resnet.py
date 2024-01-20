""" Basically ported from homura.vision
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import equinox
import jax
from equinox.nn._stateful import State
from jaxtyping import Array

from equinox_vision.models.utils import conv1x1, conv3x3


class _StateIdentity(equinox.nn.Identity):
    def __call__(self, x: Array, state: State = None, *, key=None) -> tuple[Array, State]:
        return x, state


class _StateSequential(equinox.nn.Sequential):
    def __call__(self, x: Array, state: State = None, *, key: jax.random.PRNGKeyArray = None) -> tuple[Array, State]:

        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jax.random.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            x, state = layer(x, state, key=key)
        return x, state


class BasicBlock(equinox.Module):
    conv1: equinox.Module
    conv2: equinox.Module
    norm1: equinox.Module
    norm2: equinox.Module
    downsample_conv: equinox.Module
    downsample_norm: equinox.Module
    act: equinox.Module = equinox.field(static=True)
    preact: bool = equinox.field(static=True)

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Callable[[int], equinox.Module] | None,
                 act: equinox.Module,
                 preact: bool = False,
                 *,
                 key: jax.random.PRNGKeyArray
                 ):
        key0, key1, key2 = jax.random.split(key, 3)
        channels = int(channels * (width_per_group / 16)) * groups
        use_bias = norm is None

        self.conv1 = conv3x3(in_channels, channels, stride, use_bias=use_bias, key=key0)
        self.conv2 = conv3x3(channels, channels, use_bias=use_bias, key=key1)
        self.norm1 = _StateIdentity() if norm is None else norm(in_channels if preact else channels)
        self.norm2 = _StateIdentity() if norm is None else norm(channels)

        self.downsample_conv = equinox.nn.Identity()
        self.downsample_norm = _StateIdentity()
        if in_channels != channels:
            self.downsample_conv = conv1x1(in_channels, channels, stride=stride, use_bias=use_bias, key=key2)
            self.downsample_norm = _StateIdentity() if preact else norm(channels)

        self.act = act
        self.preact = preact

    def __call__(self,
                 x: Array,
                 state: Array | None = None,
                 *,
                 key: jax.random.PRNGKeyArray = None
                 ) -> tuple[Array, Array]:
        if self.preact:
            out, state = self.norm1(x, state)
            out = self.conv1(self.act(out))
            out, state = self.norm2(out, state)
            out = self.conv2(self.act(out))
        else:
            out = self.conv1(x)
            out, state = self.norm1(out, state)
            out = self.conv2(self.act(out))
            out, state = self.norm2(out, state)
        res = self.downsample_conv(x)
        res, state = self.downsample_norm(res, state)
        out = out + res
        out = out if self.preact else self.act(out)
        return out, state


class ResNet(equinox.Module):
    norm: equinox.Module
    pool: equinox.nn.AdaptiveMaxPool2d
    fc: equinox.Module
    conv_layers: tuple[equinox.Module, ...]
    act: equinox.Module = equinox.field(static=True)
    preact: bool = equinox.field(static=True)
    return_state: bool = equinox.field(static=True)

    def __init__(self,
                 block: type[BasicBlock],
                 norm: Callable[[int], equinox.Module] | None,
                 num_classes: int,
                 layer_depth: int,
                 width: int = 16,
                 widen_factor: int = 1,
                 in_channels: int = 3,
                 groups: int = 1,
                 width_per_group: int = 16,
                 act: Callable[[Array], Array] = jax.nn.relu,
                 preact: bool = False,
                 return_state: bool = False,
                 *,
                 key: jax.random.PRNGKeyArray
                 ):
        expansion = 1
        act = equinox.nn.Lambda(act)
        self.act = act
        self.preact = preact
        self.return_state = isinstance(norm(width), equinox.nn.BatchNorm) or return_state

        key, key0, key1 = jax.random.split(key, 3)
        conv = conv3x3(in_channels, width, stride=1, use_bias=norm is None, key=key0)
        self.pool = equinox.nn.AdaptiveMaxPool2d(1)
        self.norm = norm(4 * width * expansion * widen_factor if preact else width)
        self.fc = equinox.nn.Linear(4 * width * expansion * widen_factor, num_classes, key=key1)

        def _make_layer(key: jax.random.PRNGKeyArray, in_planes: int, planes: int, stride: int
                        ) -> tuple[equinox.Module, int]:
            layers = []
            for i in range(layer_depth):
                key, key0 = jax.random.split(key)
                layers.append(block(in_planes, planes, stride if i == 0 else 1, groups, width_per_group, norm, act,
                                    preact=preact, key=key0))
                if i == 0:
                    in_planes = planes * expansion
            return _StateSequential(layers), in_planes

        key, key0, key1, key2 = jax.random.split(key, 4)
        layer1, in_plane = _make_layer(key0, width, width * widen_factor, stride=1)
        layer2, in_plane = _make_layer(key1, in_plane, width * 2 * widen_factor, stride=2)
        layer3, in_plane = _make_layer(key2, in_plane, width * 4 * widen_factor, stride=2)

        conv_layers = (conv, layer1, layer2, layer3)

        # initialization, see https://github.com/patrick-kidger/equinox/issues/179

        def kaiming_normal(w, k):
            fan_out = w.shape[0] * math.prod(w.shape[2:])
            gain = math.sqrt(2)
            std = gain / math.sqrt(fan_out)
            return jax.random.normal(k, w.shape) * std

        is_conv = lambda x: isinstance(x, equinox.nn.Conv2d)
        get_conv_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_conv) if is_conv(x)]
        conv_weights = get_conv_weights(conv_layers)
        new_weights = [kaiming_normal(w, k) for w, k in zip(conv_weights, jax.random.split(key, len(conv_weights)))]
        self.conv_layers = equinox.tree_at(get_conv_weights, conv_layers, new_weights)

    def train(self) -> ResNet:
        return equinox.tree_inference(self, False)

    def eval(self) -> ResNet:
        return equinox.tree_inference(self, True)

    def __call__(self,
                 x: Array,
                 state: Array = None,
                 *,
                 key: jax.random.PRNGKeyArray = None
                 ) -> Array | tuple[Array, Array]:
        x = self.conv_layers[0](x)
        if not self.preact:
            x, state = self.norm(x, state)
            x = self.act(x)
        for layer in self.conv_layers[1:]:
            x, state = layer(x, state)
        if self.preact:
            x, state = self.norm(x, state)
            x = self.act(x)
        x = self.pool(x)
        x = self.fc(x.reshape(-1))
        if self.return_state:
            return x, state
        return x


def batch_norm(num_channels: int,
               axis_name: str = 'batch',
               momentum: float = 0.9,  # same as PyTorch's default,
               ) -> equinox.Module:
    warnings.warn("As of equinox==0.10.10 and equinox_vision==0.0.3, batch_norm introduces significant performance "
                  "drop, especially for wrn28_2. Please use it with careful consideration.")
    return equinox.nn.BatchNorm(num_channels, axis_name=axis_name, momentum=momentum)


def resnet(key: jax.random.PRNGKeyArray,
           num_classes: int,
           depth: int,
           in_channels: int = 3,
           norm: Callable[[int], equinox.Module] | None = batch_norm,
           act: Callable[[Array], Array] = jax.nn.relu,
           **kwargs
           ) -> ResNet:
    assert (depth - 2) % 6 == 0
    layer_depth = (depth - 2) // 6
    return ResNet(BasicBlock, norm, num_classes, layer_depth, in_channels=in_channels, act=act, key=key, **kwargs)


def wide_resnet(key: jax.random.PRNGKeyArray,
                num_classes: int,
                depth: int,
                widen_factor: int,
                in_channels: int = 3,
                norm: Callable[[int], equinox.Module] | None = batch_norm,
                act: Callable[[Array], Array] = jax.nn.relu,
                **kwargs
                ) -> ResNet:
    assert (depth - 4) % 6 == 0
    layer_depth = (depth - 4) // 6
    return ResNet(BasicBlock, norm, num_classes, layer_depth, in_channels=in_channels,
                  widen_factor=widen_factor, act=act, preact=True, key=key, **kwargs)


def resnet20(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3,
             **kwargs,
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 20, in_channels, **kwargs)


def resnet32(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3,
             **kwargs,
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 32, in_channels, **kwargs)


def resnet56(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3,
             **kwargs,
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 56, in_channels, **kwargs)


def wrn16_8(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3,
            **kwargs,
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 16, 8, in_channels, **kwargs)


def wrn28_2(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3,
            **kwargs,
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 28, 2, in_channels, **kwargs)


def wrn28_10(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3,
             **kwargs,
             ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 28, 10, in_channels, **kwargs)


def wrn40_2(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3,
            **kwargs,
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 40, 2, in_channels, **kwargs)


def group_norm(num_channels: int,
               num_groups: int = 8):
    return equinox.nn.GroupNorm(num_groups, num_channels)


def resnet20_gn(key: jax.random.PRNGKeyArray,
                num_classes: int = 10,
                in_channels: int = 3,
                **kwargs,
                ) -> ResNet:
    """ ResNet by He+16 with GroupNorm
    """
    return resnet(key, num_classes, 20, in_channels, norm=group_norm, **kwargs)


def resnet56_gn(key: jax.random.PRNGKeyArray,
                num_classes: int = 10,
                in_channels: int = 3,
                **kwargs,
                ) -> ResNet:
    """ ResNet by He+16 with GroupNorm
    """
    return resnet(key, num_classes, 56, in_channels, norm=group_norm, **kwargs)


def wrn28_2_gn(key: jax.random.PRNGKeyArray,
               num_classes: int = 10,
               in_channels: int = 3,
               **kwargs,
               ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17 with GroupNorm
    """
    return wide_resnet(key, num_classes, 28, 2, in_channels, norm=group_norm, **kwargs)


def wrn40_2_gn(key: jax.random.PRNGKeyArray,
               num_classes: int = 10,
               in_channels: int = 3,
               **kwargs
               ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17 with GroupNorm
    """
    return wide_resnet(key, num_classes, 40, 2, in_channels, norm=group_norm, **kwargs)


def wrn28_10_gn(key: jax.random.PRNGKeyArray,
                num_classes: int = 10,
                in_channels: int = 3,
                **kwargs,
                ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17 with GroupNorm
    """
    return wide_resnet(key, num_classes, 28, 10, in_channels, norm=group_norm, **kwargs)
