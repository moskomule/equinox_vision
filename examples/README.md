# Examples

## [cifar10.py](./cifar10.py)

This example trains a WideResNet 28-2 with GroupNorm for approximately 200 epochs.
I found that BatchNorm significantly slows down training (~10x), probably because the state handling in equinox is suboptimal.