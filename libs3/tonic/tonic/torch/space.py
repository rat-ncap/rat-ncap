import collections
import functools
import typing as T

import gymnasium.spaces
import gymnasium.spaces.utils
import numpy as np
import torch


# ==================================================================================================
# Packing

V = T.TypeVar('V')

def pack_space(space: gymnasium.spaces.Space[V]) -> gymnasium.spaces.Box:
  space_flat = gymnasium.spaces.utils.flatten_space(space)
  assert isinstance(space_flat, gymnasium.spaces.Box)
  return space_flat

def pack_ndarrays(space: gymnasium.spaces.Space[V], value: V) -> np.ndarray:
  value_flat = gymnasium.spaces.utils.flatten(space, value)
  assert isinstance(value_flat, np.ndarray)
  return value_flat

def unpack_ndarrays(space: gymnasium.spaces.Space[V], flat: np.ndarray) -> V:
  return gymnasium.spaces.utils.unflatten(space, flat)

@functools.singledispatch
def pack_tensors(space: gymnasium.spaces.Space[V], value: V) -> torch.Tensor:
  """Pack a pytree of tensors (from  `Space`) into a single 1D tensor (into `Box`).
  
  Args:
    space: Structured space.
    value: Structured pytree of tensors to pack.
    
  Returns:
    Packed 1D tensor.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@pack_tensors.register(gymnasium.spaces.Box)
def _pack_tensors_box(space: gymnasium.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.flatten(start_dim=len(value.shape) - len(space.shape))

@pack_tensors.register(gymnasium.spaces.Dict)
def _pack_tensors_dict(space: gymnasium.spaces.Dict, value: T.Mapping[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat([pack_tensors(sp, value[key]) for key, sp in space.spaces.items()], dim=-1)

@functools.singledispatch
def unpack_tensors(space: gymnasium.spaces.Space[V], value: torch.Tensor) -> V:
  """Unpack a single 1D tensor (from `Box`) into a pytree of tensors (into `Space`).
  
  Args:
    space: Structured space.
    value: Packed 1D tensor to unpack.
    
  Returns:
    Unpacked pytree of tensors.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@unpack_tensors.register(gymnasium.spaces.Box)
def _unpack_tensors_box(space: gymnasium.spaces.Box, value: torch.Tensor) -> torch.Tensor:
  return value.reshape(list(value.shape)[:len(value.shape) - len(space.shape)] + list(space.shape))

@unpack_tensors.register(gymnasium.spaces.Dict)
def _unpack_tensors_dict(space: gymnasium.spaces.Dict, value: torch.Tensor) -> collections.OrderedDict[str, T.Any]:
  chunks = torch.split(value, [gymnasium.spaces.utils.flatdim(sp) for sp in space.spaces.values()], dim=-1)
  return collections.OrderedDict([
    (key, unpack_tensors(sp, chunk)) for chunk, (key, sp) in zip(chunks, space.spaces.items())
  ])


# ==================================================================================================
# Sampling

@functools.singledispatch
def sample_tensors(
  space: gymnasium.spaces.Space[V],
  *,
  shape: tuple[int, ...] = (),
  device: torch.device | None = None,
  rng: torch.Generator | None = None,
) -> V:
  """Sample tensors from a gymnasium space.
  
  Args:
    space: Structured space.
    shape: Batch shape of sampled tensors

  Returns:
    Sampled pytree of tensors.
  """
  raise NotImplementedError(f'Unknown space: `{space}`')

@sample_tensors.register(gymnasium.spaces.Box)
def _sample_tensors_box(
  space: gymnasium.spaces.Box,
  *,
  shape: tuple[int, ...] = (),
  device: torch.device | None = None,
  rng: torch.Generator | None = None,
) -> torch.Tensor:
  low = torch.as_tensor(space.low, dtype=torch.float32, device=device)
  high = torch.as_tensor(space.high, dtype=torch.float32, device=device)
  return low + (high - low) * torch.rand(shape + space.shape, generator=rng, device=device)
  
@sample_tensors.register(gymnasium.spaces.Dict)
def _sample_tensors_dict(
  space: gymnasium.spaces.Dict,
  *,
  shape: tuple[int, ...] = (),
  device: torch.device | None = None,
  rng: torch.Generator | None = None,
) -> collections.OrderedDict[str, T.Any]:
  return collections.OrderedDict([
    (key, sample_tensors(sp, shape=shape, device=device, rng=rng)) for key, sp in space.spaces.items()
  ])
