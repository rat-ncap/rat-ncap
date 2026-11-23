import collections
import typing as T

import gymnasium
import torch


Observation = torch.Tensor | T.OrderedDict[str, torch.Tensor]
ObservationSpace = gymnasium.spaces.Box | gymnasium.spaces.Dict
Action = torch.Tensor | dict[str, torch.Tensor]
ActionSpace = gymnasium.spaces.Box | gymnasium.spaces.Dict


class Env(gymnasium.Env[Observation, Action]):
  def initialize(self, rng: int):
    raise NotImplementedError()


class VectorEnv(gymnasium.vector.VectorEnv[Observation, Action, torch.Tensor]):
  def initialize(self, rng: torch.Tensor):
    raise NotImplementedError()


NestedTensorDict = T.Mapping[str, 'torch.Tensor | NestedTensorDict']
FlatTensorDict = T.Mapping[str, torch.Tensor]


def pack_tensors(
  tensors: dict[str, torch.Tensor],
  *,
  grad_only: bool = False,
  batch_dims: int = 0,
) -> torch.Tensor:
  """Flattens a dict of tensors into a single tensor. Assumed to all (1) be on the same device,
  (2) have the same number of batch_dims."""
  vecs = []
  for tensor in tensors.values():
    if grad_only and not tensor.requires_grad: continue
    # Keep the first `batch_dims` of shape and flatten the rest.
    vecs.append(tensor.view(*tensor.shape[:batch_dims], -1))
  return torch.cat(vecs, dim=-1)


def flatten_tensors(value: NestedTensorDict) -> FlatTensorDict:
  out = collections.OrderedDict()
  def recurse(prefix: str, node: NestedTensorDict):
    for k, v in node.items():
      if isinstance(v, dict):
        recurse(f'{prefix}{k}/', v)
      else:
        # Could be jax.Array or jax.TracedArray.
        out[f'{prefix}{k}'] = v
  recurse('', value)
  return out


@T.overload
def stack_tensors(values: T.Sequence[torch.Tensor]) -> torch.Tensor: ...
@T.overload
def stack_tensors(values: T.Sequence[FlatTensorDict]) -> FlatTensorDict: ...
@T.overload
def stack_tensors(values: T.Sequence[NestedTensorDict]) -> NestedTensorDict: ...
def stack_tensors(
  values: T.Sequence[torch.Tensor] | T.Sequence[FlatTensorDict] | T.Sequence[Observation] | T.Sequence[NestedTensorDict]
) -> torch.Tensor | FlatTensorDict | Observation | NestedTensorDict:
  """Stacks a list of dicts of tensors into a single dict of tensors."""
  is_tensor = [isinstance(v, torch.Tensor) for v in values]
  if not any(is_tensor):
    return collections.OrderedDict({
      key: stack_tensors([value[key] for value in values]) for key in values[0].keys()  # type: ignore
    })
  if all(is_tensor):
    return torch.stack(values)  # type: ignore
  assert False, 'Cannot stack mismatched dict shapes.'


def tile_tensors(tensors: FlatTensorDict, count: int) -> FlatTensorDict:
  """Tiles a dict of tensors along the first dimension."""
  tiled = {}
  batch_size = None
  for k, v in tensors.items():
    if batch_size is None:
      batch_size = v.shape[0]
    elif v.shape[0] != batch_size:
      raise ValueError(
        f'All tensors must have the same batch size, got {v.shape[0]} for key {k} and {batch_size} '
        'for previous keys.'
      )
    tiled[k] = v.repeat(count, *[1] * (v.ndim - 1))
  return tiled
