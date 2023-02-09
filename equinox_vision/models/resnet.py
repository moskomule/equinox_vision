""" Basically ported from homura.vision
"""
from __future__ import annotations

import math
from collections.abc import Callable

import equinox
import jax
from jax import numpy as jnp
from jaxtyping import Array

from equinox_vision.models.utils import conv1x1, conv3x3


class BasicBlock(equinox.Module):
    conv1: equinox.Module
    conv2: equinox.Module
    downsample: equinox.Module
    act: equinox.Module = equinox.static_field()
    preact: bool = equinox.static_field()

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
        if preact:
            self.conv1 = equinox.nn.Sequential([norm(in_channels) or equinox.nn.Identity(), act,
                                                conv3x3(in_channels, channels, stride, use_bias=use_bias, key=key0)])
            self.conv2 = equinox.nn.Sequential([norm(channels) or equinox.nn.Identity(), act,
                                                conv3x3(channels, channels, use_bias=use_bias, key=key1)])
        else:
            self.conv1 = equinox.nn.Sequential([conv3x3(in_channels, channels, stride, use_bias=use_bias, key=key0),
                                                norm(channels) or equinox.nn.Identity(), act])
            self.conv2 = equinox.nn.Sequential([conv3x3(channels, channels, use_bias=use_bias, key=key1),
                                                norm(channels) or equinox.nn.Identity()])
        self.downsample = (equinox.nn.Identity()
                           if in_channels == channels else
                           equinox.nn.Sequential([conv1x1(in_channels, channels, stride=stride,
                                                          use_bias=use_bias, key=key2),
                                                  norm(channels)]))
        self.act = act
        self.preact = preact

    def __call__(self,
                 x: Array,
                 *,
                 key: jax.random.PRNGKeyArray = None
                 ) -> Array:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.downsample(x)
        out = out if self.preact else self.act(out)
        return out


class ResNet(equinox.nn.Sequential):

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
                 *,
                 key: jax.random.PRNGKeyArray
                 ):
        expansion = 1
        act = equinox.nn.Lambda(act)
        key, key0, key1 = jax.random.split(key, 3)
        conv = conv3x3(in_channels, width, stride=1, use_bias=norm is None, key=key0)
        post_conv = equinox.nn.Identity() if preact else equinox.nn.Sequential([norm(width), equinox.nn.Lambda(act)])
        pool = equinox.nn.AdaptiveMaxPool2d(1)
        pre_pool = (equinox.nn.Sequential([norm(4 * width * expansion * widen_factor), equinox.nn.Lambda(act)])
                    if preact else equinox.nn.Identity())

        fc = equinox.nn.Linear(4 * width * expansion * widen_factor, num_classes, key=key1)

        def _make_layer(in_planes: int, planes: int, stride: int) -> tuple[equinox.Module, int]:
            layers = []
            for i in range(layer_depth):
                key0, key1 = jax.random.split(key)
                layers.append(block(in_planes, planes, stride if i == 0 else 1, groups, width_per_group, norm, act,
                                    preact=preact, key=key0))
                if i == 0:
                    in_planes = planes * expansion
            return equinox.nn.Sequential(layers), in_planes

        layer1, in_plane = _make_layer(width, width * widen_factor, stride=1)
        layer2, in_plane = _make_layer(in_plane, width * 2 * widen_factor, stride=2)
        layer3, in_plane = _make_layer(in_plane, width * 4 * widen_factor, stride=2)

        layers = (conv, post_conv, layer1, layer2, layer3, pre_pool, pool,
                  equinox.nn.Lambda(lambda x: jnp.reshape(x, (-1,))), fc)

        # initialization, see https://github.com/patrick-kidger/equinox/issues/179

        def kaiming_normal(w, k):
            fan_out = w.shape[0] * math.prod(w.shape[2:])
            gain = math.sqrt(2)
            std = gain / math.sqrt(fan_out)
            return jax.random.normal(k, w.shape) * std

        is_conv = lambda x: isinstance(x, equinox.nn.Conv2d)
        get_conv_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_conv) if is_conv(x)]
        conv_weights = get_conv_weights(layers)
        new_weights = [kaiming_normal(w, k) for w, k in zip(conv_weights, jax.random.split(key, len(conv_weights)))]
        layers = equinox.tree_at(get_conv_weights, layers, new_weights)

        super().__init__(layers)

    def train(self) -> ResNet:
        return equinox.tree_inference(self, False)

    def eval(self) -> ResNet:
        return equinox.tree_inference(self, True)


def batch_norm(num_channels: int,
               axis_name: str = 'batch',
               momentum: float = 0.9,  # same as PyTorch's default,
               ) -> equinox.Module:
    return equinox.experimental.BatchNorm(num_channels, axis_name=axis_name, momentum=momentum)


def resnet(key: jax.random.PRNGKeyArray,
           num_classes: int,
           depth: int,
           in_channels: int = 3,
           norm: Callable[[int], equinox.Module] | None = batch_norm,
           act: Callable[[Array], Array] = jax.nn.relu,
           **kwargs
           ) -> ResNet:
    "resnet-{depth}"
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
    "wideresnet-{depth}-{widen_factor}"
    assert (depth - 4) % 6 == 0
    layer_depth = (depth - 4) // 6
    return ResNet(BasicBlock, norm, num_classes, layer_depth, in_channels=in_channels,
                  widen_factor=widen_factor, act=act, preact=True, key=key, **kwargs)


def resnet20(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 20, in_channels, )


def resnet32(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 32, in_channels, )


def resnet56(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(key, num_classes, 56, in_channels, )


def wrn16_8(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 16, 8, in_channels, )


def wrn28_2(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 28, 2, in_channels, )


def wrn28_10(key: jax.random.PRNGKeyArray,
             num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 28, 10, in_channels)


def wrn40_2(key: jax.random.PRNGKeyArray,
            num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(key, num_classes, 40, 2, in_channels)
