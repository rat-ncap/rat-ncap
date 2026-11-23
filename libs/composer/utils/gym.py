import collections
import typing as T

import equinox as eqx
import gymnasium.spaces
import gymnasium.spaces.utils
import gymnasium.vector
import jax
import numpy as np
import numpy.typing as npt
import torch

from .. import core, mjx_env
from ..utils import torch as cxt

# ==================================================================================================
# Conversion


class Box(gymnasium.spaces.Box):
  """A gymnasium `Box` that keeps a composer `core.Box`'s name, description and axis labels."""

  def __init__(
    self,
    low: T.SupportsFloat | npt.NDArray[T.Any],
    high: T.SupportsFloat | npt.NDArray[T.Any],
    shape: T.Sequence[int] | None = None,
    dtype: type[np.floating[T.Any]] | type[np.integer[T.Any]] = np.float32,
    seed: int | np.random.Generator | None = None,
    name: str = '',
    desc: str = '',
    labels: T.Sequence[str | T.Sequence[str]] | None = None,
  ):
    """Initializes the box.

    Args:
      low: Lower bounds, as in `gymnasium.spaces.Box`.
      high: Upper bounds, as in `gymnasium.spaces.Box`.
      shape: Shape of the values, as in `gymnasium.spaces.Box`.
      dtype: Dtype of the values, as in `gymnasium.spaces.Box`.
      seed: Seed of the space's sampler, as in `gymnasium.spaces.Box`.
      name: Short human-readable name.
      desc: Longer human-readable description.
      labels: Axis labels, as in `core.Box`.
    """
    super().__init__(low, high, shape, dtype, seed)
    self.name = name
    self.desc = desc
    self.labels = labels and tuple([l if isinstance(l, str) else tuple(l) for l in labels])


@T.overload
def to_gym_space(space: core.Box) -> gymnasium.spaces.Box: ...
@T.overload
def to_gym_space(space: core.Dict) -> gymnasium.spaces.Dict: ...
def to_gym_space(space: core.Space) -> gymnasium.spaces.Space:
  """Converts a composer space to the equivalent gymnasium space.

  A `core.Box` becomes a `Box` (keeping its name, description and labels), and a `core.Dict` a
  `gymnasium.spaces.Dict` with the same key order.

  Raises:
    NotImplementedError: For other space types.
  """
  match space:
    case core.Box():
      return Box(
        low=np.asarray(space.low),
        high=np.asarray(space.high),
        shape=space.shape,
        dtype=space.dtype,
        name=space.name,
        desc=space.desc,
        labels=space.labels,
      )
    case core.Dict():
      return gymnasium.spaces.Dict(
        collections.OrderedDict([(key, to_gym_space(value)) for key, value in space.items()])
      )
    case _:
      raise NotImplementedError(f'Unsupported space type: {type(space)}')


# ==================================================================================================
# Flattening


def pack_space(space: gymnasium.spaces.Space) -> gymnasium.spaces.Box:
  """Returns the flat `Box` space of `space`'s values, as `gymnasium.spaces.utils.flatten_space`."""
  space_flat = gymnasium.spaces.utils.flatten_space(space)
  assert isinstance(space_flat, gymnasium.spaces.Box)
  return space_flat


def pack_ndarrays[V](space: gymnasium.spaces.Space[V], value: V) -> np.ndarray:
  """Returns `value` of `space` as one flat array, as `gymnasium.spaces.utils.flatten`."""
  value_flat = gymnasium.spaces.utils.flatten(space, value)
  assert isinstance(value_flat, np.ndarray)
  return value_flat


def unpack_ndarrays[V](space: gymnasium.spaces.Space[V], flat: np.ndarray) -> V:
  """Returns the value of `space` packed into `flat` by `pack_ndarrays`."""
  return gymnasium.spaces.utils.unflatten(space, flat)


def pack_tensors(
  space: gymnasium.spaces.Space,
  value: torch.Tensor | T.Mapping[str, T.Any],
) -> torch.Tensor:
  """Flattens a batch of values of `space` into one tensor, for example to feed a network.

  Args:
    space: A `Box`, or a `Dict` of them (nested).
    value: Values of `space` with any leading batch axes, `[*batch, *shape]` per box.

  Returns:
    The values flattened and concatenated in key order, `[*batch, flatdim(space)]`.

  Raises:
    NotImplementedError: For other space types.
  """
  match space:
    case gymnasium.spaces.Box():
      value = T.cast(torch.Tensor, value)
      return value.flatten(start_dim=value.ndim - len(space.shape))
    case gymnasium.spaces.Dict():
      value = T.cast(T.Mapping[str, T.Any], value)
      return torch.cat([pack_tensors(sp, value[key]) for key, sp in space.spaces.items()], dim=-1)
    case _:
      raise NotImplementedError(f'Unknown space: `{space}`')


def unpack_tensors(
  space: gymnasium.spaces.Space,
  value: torch.Tensor,
) -> torch.Tensor | collections.OrderedDict[str, T.Any]:
  """Unflattens a batch of values of `space` packed by `pack_tensors`.

  Args:
    space: A `Box`, or a `Dict` of them (nested).
    value: Flat values with any leading batch axes, `[*batch, flatdim(space)]`.

  Returns:
    The values with the structure of `space`, `[*batch, *shape]` per box.

  Raises:
    NotImplementedError: For other space types.
  """
  match space:
    case gymnasium.spaces.Box():
      return value.reshape(value.shape[:-1] + space.shape)
    case gymnasium.spaces.Dict():
      sizes = [gymnasium.spaces.utils.flatdim(sp) for sp in space.spaces.values()]
      chunks = torch.split(value, sizes, dim=-1)
      return collections.OrderedDict([
        (key, unpack_tensors(sp, chunk)) for chunk, (key, sp) in zip(chunks, space.spaces.items())
      ])
    case _:
      raise NotImplementedError(f'Unknown space: `{space}`')


# ==================================================================================================
# Gymnasium adapter


class ComposerEnv(gymnasium.vector.VectorEnv):
  """A composer `MjxEnv` as a gymnasium `VectorEnv`, with torch tensors in and out (for Tonic).

  Call `initialize` before anything else. Unlike gymnasium's convention, `observation_space` and
  `action_space` are the spaces of one world (gymnasium's `single_observation_space` and
  `single_action_space`), which is what Tonic reads.

  Attributes:
    env: The wrapped environment.
    autoreset: Whether `step` resets worlds whose episode ended, in the same step.
    rng: Random key of each world, `[vectorize]`.
    state: The current state of every world.
    subtask: The subtask passed to every reset, as set by `reset`'s `options`.
  """
  env: mjx_env.MjxEnv
  autoreset: bool
  rng: jax.Array
  state: mjx_env.MjxEnvState
  subtask: mjx_env.Subtask

  def __init__(
    self,
    env: mjx_env.MjxEnv,
    autoreset: bool = True,
  ):
    """Initializes the adapter.

    Args:
      env: The environment.
      autoreset: Whether `step` resets worlds whose episode ended, in the same step ("same-step"
        autoreset, https://farama.org/Vector-Autoreset-Mode).
    """
    self.env = env
    self.autoreset = autoreset
    self.subtask = {}

  def initialize(self, rng: torch.Tensor):
    """Initializes the worlds and the spaces.

    Args:
      rng: An integer seed per world, `[vectorize]`.
    """
    self.rng = jax.vmap(jax.random.key)(cxt.torch_to_jax(rng))
    self.state = self.env.initialize(self.rng)
    self.observation_space = to_gym_space(self.env.observation_space(self.state).unvectorize())
    self.action_space = to_gym_space(self.env.action_space(self.state).unvectorize())
    self.env_reset = eqx.filter_jit(self.env.reset)
    self.env_step = eqx.filter_jit(self.env.step)

  @T.override
  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, T.Any] | None = None,
  ) -> tuple[cxt.NestedTensorDict, dict[str, T.Any]]:
    """Starts new episodes in all worlds.

    Args:
      seed: Seed of the episodes' random keys. A fresh one is drawn if `None`.
      options: May hold `subtask`, the subtask of each world (as in `MjxEnv.reset`), which is also
        used by later autoresets.

    Returns:
      The observations of every world, and an empty info dict.
    """
    if seed is None:
      seed = int(np.random.default_rng().integers(2**31))
    options = options or {}
    self.subtask = cxt.torch_to_jax(options.get('subtask', {}))

    rng = jax.random.split(jax.random.key(seed), self.state.vectorize)
    step = self.env_reset(self.state, rng, subtask=self.subtask)
    self.state = step.state

    observations = cxt.jax_to_torch(step.observation)
    return observations, {}

  @T.override
  def step(
    self,
    actions: dict[str, torch.Tensor],
  ) -> tuple[
    torch.Tensor | cxt.NestedTensorDict,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, T.Any],
  ]:
    """Steps every world with `actions`, then autoresets the worlds whose episode ended.

    Args:
      actions: Action of each world, `[vectorize, ...]`.

    Returns:
      `(observations, rewards, terminations, truncations, info)` of the step. `observations` are
      the step's own (for ended episodes, their final observation). `info` holds
      `new_observations`, the observations to act on next (after autoreset), and each metric as
      `metrics/<name>`.
    """
    step = self.env_step(self.state, cxt.torch_to_jax(actions))
    observations = cxt.jax_to_torch(step.observation)
    rewards = cxt.jax_to_torch(step.reward)
    terminations = cxt.jax_to_torch(step.terminated)
    truncations = cxt.jax_to_torch(step.truncated)
    if self.autoreset:
      reset = step.terminated | step.truncated
      step = self.env_reset(step.state, self.rng, subtask=self.subtask, reset=(reset, step))
    self.state = step.state
    new_observations = cxt.jax_to_torch(step.observation)

    info = {
      'new_observations': new_observations,
    } | {
      f'metrics/{key}': cxt.jax_to_torch(value) for key, value in step.metrics.items()
    }
    return observations, rewards, terminations, truncations, info
