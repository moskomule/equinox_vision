import jax
from equinox import nn


def conv3x3(in_channels: int,
            out_channels: int,
            stride: int = 1,
            use_bias: bool = False,
            groups: int = 1,
            *,
            key: Array
            ) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, use_bias=use_bias,
                     groups=groups, key=key)


def conv1x1(in_channels: int,
            out_channels: int,
            stride=1,
            use_bias: bool = False,
            *,
            key: Array
            ) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, use_bias=use_bias, key=key)
