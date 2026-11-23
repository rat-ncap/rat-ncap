"""This module facilitates controlled sampling of noise."""
import typing as T

import torch


def multiseed(lengths: T.Sequence[int], seeds: T.Sequence[int]) -> int:
  """Combine multiple integer `seeds` into a single composite integer seed.

  Each input seed is allocated its corresponding `length` (in decimal digits) in the composite seed,
  and larger seeds are shortened via the modulo operator, so seeds that differ by a multiple of
  `10**length` collide (e.g. with a length of 5, epoch 100000 gives the same seed as epoch 0). The
  composite seed is prefixed by 1 to ensure consistent zero-padding, so it has `1 + sum(lengths)`
  digits. For example:
  ```
  multiseed(lengths=(2, 3, 4), seeds=(2, 333, 10004))
  >>> 1023330004  # 1-02-333-0004
  ```

  Args:
    lengths: Number of decimal digits of each seed.
    seeds: The non-negative seeds, one per length.

  Returns:
    The composite seed.
  """
  assert len(lengths) == len(seeds) != 0
  seed = 1
  for i in range(len(lengths)):
    seed = seed * 10**lengths[i] + seeds[i] % 10**lengths[i]
  return seed


def gaussian(size: T.Sequence[int], seed: int) -> torch.Tensor:
  """Generate a block of Gaussian noise with a mean of 0 and standard deviation of 1, on the
  default device. The noise of a seed depends on the device type, as CPU and CUDA generators
  produce different streams."""
  g = torch.Generator(torch.get_default_device()).manual_seed(seed)
  return torch.randn(size, generator=g)


def bernoulli(size: T.Sequence[int], seed: int) -> torch.Tensor:
  """Generate a block of Bernoulli noise with values in {-1, +1} (int64), on the default device."""
  g = torch.Generator(torch.get_default_device()).manual_seed(seed)
  return torch.randint(0, 2, size, generator=g) * 2 - 1


def uniform(size: T.Sequence[int], seed: int) -> torch.Tensor:
  """Generate a block of uniform noise with values in [-1, +1], on the default device."""
  g = torch.Generator(torch.get_default_device()).manual_seed(seed)
  return torch.rand(size, generator=g) * 2 - 1
