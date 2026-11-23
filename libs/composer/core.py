import abc
import collections
import dataclasses
import itertools
import math as mathpy
import types
import typing as T

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import wadler_lindig as wl

# ==================================================================================================
# Spaces

SampleMode = T.Literal['random', 'zero', 'min', 'max', 'mid']
"""How `Space.sample` picks a value: uniformly at random, zero clipped into the bounds, the lower
bound, the upper bound, or the midpoint of the bounds."""

class Space[Value](eqx.Module):
  """The set of values an action or observation can take.

  Attributes:
    name: Short human-readable name.
    desc: Longer human-readable description.
  """
  name: str = eqx.field(static=True)
  desc: str = eqx.field(static=True)

  @abc.abstractmethod
  def sample(self, rng: jax.Array, mode: SampleMode = 'random') -> Value:
    """Returns a value in the space, chosen according to `mode`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def vectorize(self, size: int) -> 'Space':
    """Returns the space of `size` stacked values: a new leading axis of length `size`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def unvectorize(self) -> 'Space':
    """Returns the space of one value of a vectorized space: the leading axis removed."""
    raise NotImplementedError()

  @abc.abstractmethod
  def validate(self, value: Value) -> None:
    """Raises if `value` does not have the space's structure. Bounds are not checked."""
    raise NotImplementedError()


class Box(Space[jax.Array]):
  """An array of a fixed shape and dtype with elementwise bounds, optionally with labeled axes.

  Labels name the axes: each entry of `labels` is either a `str` that names the whole axis (for
  example `'joint'`) or a tuple of `str` that names each index along it (for example
  `('hip', 'knee')`). `None` means the space is unlabeled.
  """
  low: jax.Array
  high: jax.Array
  shape: tuple[int, ...] = eqx.field(static=True)
  labels: tuple[str | tuple[str, ...], ...] | None = eqx.field(static=True)
  dtype: np.dtype = eqx.field(static=True)

  def __init__(
    self,
    low: jax.Array | float,
    high: jax.Array | float,
    shape: tuple[int, ...] | None = None,
    labels: T.Sequence[str | T.Sequence[str]] | None = None,
    dtype: jt.DTypeLike = jnp.float32,
    name: str = '',
    desc: str = '',
  ):
    """Initializes the box.

    Args:
      low: Lower bounds, broadcast to `shape`.
      high: Upper bounds, broadcast to `shape`.
      shape: Shape of the values. Defaults to the broadcast shape of `low` and `high`.
      labels: One label per axis (see the class docstring), or `None` for no labels. A tuple label
        must have one name per index of its axis, and names must be unique along an axis.
      dtype: Dtype of the values and bounds.
      name: Short human-readable name.
      desc: Longer human-readable description.
    """
    low = jnp.asarray(low, dtype=dtype)
    high = jnp.asarray(high, dtype=dtype)
    if shape is None:
      shape = jnp.broadcast_shapes(low.shape, high.shape)

    if labels is not None:
      # Validate label shape.
      labels = tuple([l if isinstance(l, str) else tuple(l) for l in labels])
      assert len(labels) == len(shape) and all(
        isinstance(l, str) or len(l) == dim for l, dim in zip(labels, shape)
      ), f'Label {labels} does not match shape {shape}.'
      # Disallow duplicate labels along a given axis.
      label_axes = [
        tuple(l for l in labels if isinstance(l, str))
      ] + [l for l in labels if not isinstance(l, str)]
      assert all(len(a) == len(set(a)) for a in label_axes), (
        f'Label {labels} has duplicate names along an axis.'
      )

    low = jnp.broadcast_to(low, shape).astype(dtype)
    high = jnp.broadcast_to(high, shape).astype(dtype)
    assert low.shape == high.shape
    self.low = low
    self.high = high
    self.shape = shape
    self.labels = labels
    self.dtype = low.dtype
    self.name = name
    self.desc = desc

  @T.override
  def sample(self, rng: jax.Array, mode: SampleMode = 'random') -> jax.Array:
    """Returns a value within the bounds, chosen according to `mode` (see `SampleMode`)."""
    match mode:
      case 'random':
        return jax.random.uniform(
          rng,
          shape=self.shape,
          minval=self.low,
          maxval=self.high,
          dtype=self.dtype,
        )
      case 'zero':
        return jnp.clip(jnp.zeros(self.shape, dtype=self.dtype), self.low, self.high)
      case 'min':
        return self.low.copy()
      case 'max':
        return self.high.copy()
      case 'mid':
        return (self.low + self.high) / 2
      case _:
        raise ValueError(f'Invalid sample mode: {mode}')

  @T.override
  def vectorize(self, size: int, label: str | T.Sequence[str] | None = None) -> 'Box':
    """Returns the space of `size` stacked values, with the bounds tiled along a new leading axis.

    Args:
      size: Length of the new leading axis.
      label: Label of the new axis: a `str` naming the axis, or one name per index. Defaults to the
        first unused `vectorize<i>`. Only allowed if the space is labeled.

    Returns:
      The vectorized space.
    """
    shape = (size,) + self.shape
    assert label is None or self.labels is not None, (
      f'Cannot prepend label `{label}` to vectorized space if original space has no label.'
    )
    # Axis names already in use (a str label names a whole axis; a tuple names its indices).
    axis_names: set[str] = set()
    if self.labels is not None:
      axis_names = {l for l in self.labels if isinstance(l, str)}
    vectorize_label: str | tuple[str, ...]
    if label is None:
      vectorize_label = next(
        f'vectorize{i}' for i in itertools.count() if f'vectorize{i}' not in axis_names
      )
    elif isinstance(label, str):
      assert label not in axis_names, (
        f'Label `{label}` already exists in original label {self.labels}.'
      )
      vectorize_label = label
    else:
      vectorize_label = tuple(label)
      assert len(vectorize_label) == size and len(set(vectorize_label)) == size, (
        f'Label `{label}` must name each of the {size} vectorized indices exactly once.'
      )

    return dataclasses.replace(
      self,
      low=jnp.broadcast_to(self.low, shape),
      high=jnp.broadcast_to(self.high, shape),
      shape=shape,
      labels=None if self.labels is None else (vectorize_label,) + self.labels,
    )

  @T.override
  def unvectorize(self) -> 'Box':
    """Returns the space of one value: the leading axis and its label removed."""
    assert len(self.shape) >= 1
    return dataclasses.replace(
      self,
      low=self.low[0],
      high=self.high[0],
      shape=self.shape[1:],
      labels=None if self.labels is None else self.labels[1:],
    )

  @T.override
  def validate(self, value: jax.Array) -> None:
    """Raises `ValueError` if `value`'s shape or dtype differ from the space's."""
    if value.shape != self.shape:
      raise ValueError(f'Shape mismatch: expected {self.shape}, got {value.shape}')
    if value.dtype != self.dtype:
      raise ValueError(f'Dtype mismatch: expected {self.dtype}, got {value.dtype}')


class Dict(Space[T.Mapping[str, jax.Array]]):
  """An ordered mapping from names to `Box` spaces, for dictionaries of arrays.

  Unlike most Equinox modules it is mutable: `__setitem__`, `__delitem__` and `update` change the
  wrapped dictionary in place. Use `copy()` or `|` for a modified copy.
  """
  spaces: collections.OrderedDict[str, Box]

  def __init__(self, spaces: T.Mapping[str, Box] | None = None, name: str = '', desc: str = ''):
    """Initializes the space.

    Args:
      spaces: The named spaces, in order. Defaults to no spaces.
      name: Short human-readable name.
      desc: Longer human-readable description.
    """
    self.spaces = collections.OrderedDict(spaces or {})
    self.name = name
    self.desc = desc

  @T.override
  def sample(self, rng: jax.Array, mode: SampleMode = 'random') -> T.Mapping[str, jax.Array]:
    """Returns one value per key, each sampled with its own split of `rng`."""
    if len(self) == 0: return collections.OrderedDict({})
    out: dict[str, jax.Array] = collections.OrderedDict()
    _, *rngs = jax.random.split(rng, len(self) + 1)
    for (k, v), r in zip(self.items(), rngs):
      out[k] = v.sample(r, mode=mode)
    return out

  @T.override
  def vectorize(self, size: int) -> 'Dict':
    """Returns the space with every entry vectorized to `size` (see `Box.vectorize`)."""
    return Dict(
      spaces=collections.OrderedDict({
        k: v.vectorize(size) for k, v in self.items()
      }),
      name=self.name,
      desc=self.desc,
    )

  @T.override
  def unvectorize(self) -> 'Dict':
    """Returns the space with every entry unvectorized (see `Box.unvectorize`)."""
    return Dict(
      spaces=collections.OrderedDict({
        k: v.unvectorize() for k, v in self.items()
      }),
      name=self.name,
      desc=self.desc,
    )

  @T.override
  def validate(self, value: T.Mapping[str, jax.Array]) -> None:
    """Raises `KeyError` if the keys differ, or `ValueError` if an entry's shape or dtype does."""
    # Validate number, and names of keys.
    if len(self) != len(value) or set(self.keys()) != set(value.keys()):
      raise KeyError(f"Key mismatch: expected space {self}, got value {value}")
    for k, sp in self.items():
      v = value[k]
      sp.validate(v)

  def __len__(self) -> int:
    """Returns the number of entries."""
    return len(self.spaces)

  def __iter__(self) -> T.Iterator[str]:
    """Iterates over the keys in order."""
    return iter(self.spaces)

  def __getitem__(self, key: str) -> Box:
    """Returns the space of `key`."""
    return self.spaces[key]

  def __setitem__(self, key: str, value: Box) -> None:
    """Sets the space of `key` in place."""
    self.spaces[key] = value

  def __delitem__(self, key: str) -> None:
    """Removes `key` in place."""
    del self.spaces[key]

  def __contains__(self, key: str) -> bool:
    """Returns whether `key` has a space."""
    return key in self.spaces

  def __or__(self, other: T.Mapping[str, Box]) -> 'Dict':
    """Returns a copy updated with `other`'s entries."""
    new_spaces = self.spaces.copy()
    new_spaces.update(other)
    return Dict(spaces=new_spaces, name=self.name, desc=self.desc)

  def keys(self) -> T.KeysView[str]:
    """Returns the keys in order."""
    return self.spaces.keys()

  def values(self) -> T.ValuesView[Box]:
    """Returns the spaces in order."""
    return self.spaces.values()

  def items(self) -> T.ItemsView[str, Box]:
    """Returns the (key, space) pairs in order."""
    return self.spaces.items()

  def get(self, key: str, default: Box | None = None) -> Box | None:
    """Returns the space of `key`, or `default` if absent."""
    return self.spaces.get(key, default)

  def copy(self) -> 'Dict':
    """Returns a shallow copy (the `Box` spaces are shared)."""
    return Dict(spaces=self.spaces.copy(), name=self.name, desc=self.desc)

  def update(self, other: 'T.Mapping[str, Box] | Dict') -> None:
    """Adds or replaces entries from `other`, in place."""
    self.spaces.update(other)


ArrayDict = collections.OrderedDict[str, jax.Array]
"""Values of a `Dict` space, in the space's key order."""


def pack_space(space: Dict, *, batch_dims: int = 0) -> Box:
  """Packs a dict of spaces into a single space. Assumed to all (1) be on the same device,
  (2) have the same number of batch_dims."""
  dtypes, lows, highs, names, descs = [], [], [], [], []
  for v in space.values():
    if isinstance(v, Box):
      batch_shape = v.shape[:batch_dims]
      size = mathpy.prod(v.shape[batch_dims:])
      dtypes.append(v.dtype)
      lows.append(v.low.reshape(*batch_shape, size))
      highs.append(v.high.reshape(*batch_shape, size))
      names.append(v.name)
      descs.append(v.desc)
    else:
      raise TypeError(f'Unsupported space type: {type(v)}')
  if len(dtypes) == 0: raise ValueError(f'No spaces to pack: {space}')
  low = jnp.concatenate(lows, axis=-1)
  high = jnp.concatenate(highs, axis=-1)
  return Box(
    low=low,
    high=high,
    dtype=low.dtype,
    name='\n'.join(names),
    desc='\n'.join(descs),
  )

def pack_value(space: Dict, value: ArrayDict, *, batch_dims: int = 0) -> jax.Array:
  """Packs a dict of arrays into a single array. Assumed to all (1) be on the same device,
  (2) have the same number of batch_dims."""
  arrays = []
  for k in space.keys():
    v = value[k]
    arrays.append(v.reshape(*v.shape[:batch_dims], -1))  # Could be jax.Array or jax.TracedArray.
  return jnp.concatenate(arrays, axis=-1)

def unpack_value(space: Dict, value: jax.Array) -> ArrayDict:
  """Unpacks a single array into a dict of arrays, according to the given space. Assumed to
  all (1) be on the same device, (2) have the same number of batch_dims."""
  out = collections.OrderedDict()
  offset = 0
  batch_dims = len(value.shape) - 1
  for k, v in space.items():
    if isinstance(v, Box):
      size = mathpy.prod(v.shape[batch_dims:])
      out[k] = value[..., offset:offset + size].reshape(v.shape)
      offset += size
    else:
      raise TypeError(f'Unsupported space type: {type(v)}')
  if offset != value.shape[-1]:
    raise ValueError(f'Incorrect vector size to unpack: {list(value.shape)}')
  return out


def tile_value(value: dict[str, jax.Array], count: int) -> ArrayDict:
  """Repeats each array `count` times along its leading axis, as whole copies.

  For a value with leading axis `[n]` the result has leading axis `[count * n]`, laid out as
  `count` consecutive blocks of the original `n` rows (population-major when `count` is the
  population). A 0-d array becomes `[count]`.
  """
  return collections.OrderedDict({
    k: jnp.tile(
      jnp.array(v),
      [count] + [1] * (len(jnp.array(v).shape) - 1),
    ) for k, v in value.items()
  })


# ==================================================================================================
# Env and Wrapper

Action = T.Mapping[str, jax.Array]  # Order doesn't matter. Mapping to be contravariant.
ActionSpace = Dict
Observation = collections.OrderedDict[str, jax.Array]
ObservationSpace = Dict
Metrics = dict[str, jax.Array]  # Order doesn't matter.


class Step[EnvState](eqx.Module):
  """The result of `Env.reset` or `Env.step`.

  Attributes:
    state: Environment state after the step.
    observation: Observation of `state`.
    terminated: Whether the episode has ended (bool). Once true, it stays true for the rest of the
      episode, and later steps have zero reward.
    truncated: Whether the episode hit its step limit (bool).
    reward: Reward of the step (float).
    metrics: Named diagnostic values of the step.
  """
  state: EnvState
  observation: Observation
  terminated: jax.Array  # dtype=bool
  truncated: jax.Array  # dtype=bool
  reward: jax.Array  # dtype=float
  metrics: Metrics


class Env[EnvState](eqx.Module):
  """A functional environment: every method takes the state and returns a new one."""

  @abc.abstractmethod
  def initialize(self, rng: jax.Array) -> EnvState:
    """Returns the initial state, before the first `reset`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def reset(self, state: EnvState, rng: jax.Array) -> Step[EnvState]:
    """Starts a new episode from `state`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, state: EnvState, action: Action) -> Step[EnvState]:
    """Advances the episode by one step with `action`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def action_space(self, state: EnvState) -> ActionSpace:
    """Returns the space of actions accepted by `step`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def observation_space(self, state: EnvState) -> ObservationSpace:
    """Returns the space of observations returned in a `Step`."""
    raise NotImplementedError()

  @property
  def unwrapped(self) -> 'Env[EnvState]':
    """The innermost environment, without wrappers."""
    return self


class Wrapper[EnvState](Env):
  """An environment that forwards every method, and unknown attributes, to `env`. Subclasses
  override the methods they change."""
  env: Env

  @T.override
  def initialize(self, rng: jax.Array) -> EnvState:
    """Forwards to the wrapped environment."""
    return self.env.initialize(rng)

  @T.override
  def reset(self, state: EnvState, rng: jax.Array) -> Step[EnvState]:
    """Forwards to the wrapped environment."""
    return self.env.reset(state, rng)

  @T.override
  def step(self, state: EnvState, action: Action) -> Step[EnvState]:
    """Forwards to the wrapped environment."""
    return self.env.step(state, action)

  @T.override
  def action_space(self, state: EnvState) -> ActionSpace:
    """Forwards to the wrapped environment."""
    return self.env.action_space(state)

  @T.override
  def observation_space(self, state: EnvState) -> ObservationSpace:
    """Forwards to the wrapped environment."""
    return self.env.observation_space(state)

  def __getattr__(self, name: str) -> T.Any:
    """Returns the attribute `name` of the wrapped environment."""
    return getattr(self.env, name)

  @property
  def unwrapped(self) -> Env[EnvState]:
    """The innermost environment, without wrappers."""
    return self.env.unwrapped


# ==================================================================================================
# State

# The type `eqx.nn.delete_init_state` puts in place of a `StateIndex.init`. Equinox does not export
# it, so it is taken from the public function's output.
_DeletedInit = type(eqx.nn.delete_init_state(eqx.nn.StateIndex(None)).init)


@jax.tree_util.register_pytree_node_class
class State:
  """Stores the state of a pytree of Equinox modules.

  Like `eqx.nn.State` it (a) acts like a dictionary mapping `eqx.nn.StateIndex` keys to state
  values by working around JAX constraints against using objects as dict keys, (b) does
  error-checking that ensures the most up-to-date version of it is used.

  Unlike `eqx.nn.State` it (a) does not require state structures/shapes/dtypes to be fixed at
  initialization, (b) adds some convenience methods for freezing and cloning the state.
  """
  def __init__(self, pytree: jt.PyTree, frozen: bool = True):
    """Initializes the state from the `init` values of every `eqx.nn.StateIndex` in `pytree`.

    Args:
      pytree: Pytree whose `StateIndex` markers were set by `initialize_state_index`.
      frozen: Whether `set` must keep each value's structure, shape and dtype.

    Raises:
      ValueError: If a `StateIndex` has a deleted `init` or a marker that is not a string.
    """
    # Note that de/serialisation depends on the ordered-ness of this dictionary,
    # between serialisation and deserialisation.
    state = {}
    leaves = jax.tree.leaves(pytree, is_leaf=lambda x: isinstance(x, eqx.nn.StateIndex))
    for leaf in leaves:
      if isinstance(leaf, eqx.nn.StateIndex):
        if isinstance(leaf.init, _DeletedInit):
          raise ValueError(f'StateIndex.init value was already deleted: {leaf}')
        if not isinstance(leaf.marker, str):
          raise ValueError(
            f'StateIndex.marker is not string: {leaf.marker}. Use `initialize_state_index()` '
            f'before constructing state.'
          )
        state[leaf.marker] = jax.tree.map(jnp.asarray, leaf.init)
    # None marks a stale state, superseded by the `State` returned from `set()` or `freeze()`.
    self._state: dict[T.Hashable, T.Any] | None = state
    self._frozen = frozen

  def get[Value](self, index: eqx.nn.StateIndex[Value]) -> Value:
    """Get the state value for the given index."""
    if self._state is None:
      raise ValueError('Attempted to get value of stale state. Use `clone()` to preserve state.')
    if type(index) is not eqx.nn.StateIndex:
      raise ValueError(f'Can only use `eqx.nn.StateIndex` as state keys: {index}')
    return self._state[index.marker]

  def set[Value](self, index: eqx.nn.StateIndex[Value], value: Value) -> 'State':
    """Set the state value for the given index."""
    if self._state is None:
      raise ValueError('Attempted to set value of stale state.')
    if type(index) is not eqx.nn.StateIndex:
      raise ValueError(f'Can only use `eqx.nn.StateIndex` as state keys: {index}')
    new_value = jax.tree.map(jnp.asarray, value)
    if self._frozen:
      old_value = self._state[index.marker]
      old_struct = jax.eval_shape(lambda: old_value)
      new_struct = jax.eval_shape(lambda: new_value)
      if eqx.tree_equal(old_struct, new_struct) is not True:
        old_repr = eqx.tree_pformat(old_value, struct_as_array=True)
        new_repr = eqx.tree_pformat(new_value, struct_as_array=True)
        raise ValueError(
          f'Attempted to change the structure/shape/dtype of a frozen state value:\n'
          f'Old value:\n{old_repr}\n'
          f'New value:\n{new_repr}'
        )
    state = self._state.copy()
    state[index.marker] = new_value
    new_self = object.__new__(State)
    new_self._state = state
    new_self._frozen = self._frozen
    self._state = None  # Invalidate the old state.
    return new_self

  def freeze(self) -> 'State':
    """Freeze the state structure/shape/dtype."""
    if self._state is None:
      raise ValueError('Attempted to freeze a stale state.')
    new_self = object.__new__(State)
    new_self._state = self._state.copy()
    new_self._frozen = True
    self._state = None  # Invalidate the old state.
    return new_self

  def clone(self) -> 'State':
    """Clone the state to preserve the current values."""
    if self._state is None:
      raise ValueError('Attempted to clone a stale state.')
    leaves, treedef = jax.tree.flatten(self)
    clone_self = jax.tree.unflatten(treedef, leaves)
    return clone_self

  def __repr__(self) -> str:
    """Returns the Equinox pretty-printed state."""
    return eqx.tree_pformat(self)

  def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
    """Returns the `wadler_lindig` document used by `eqx.tree_pformat`."""
    if self._state is None:
      return wl.TextDoc('State(~old~)')
    else:
      docs = wl.named_objs(((str(k), v) for k, v in self._state.items()), **kwargs)
      return wl.bracketed(
        begin=wl.TextDoc('State('),
        docs=docs,
        sep=wl.comma,
        end=wl.TextDoc(')'),
        indent=kwargs["indent"],
      )

  def tree_flatten(self) -> tuple[tuple, tuple]:
    """Flatten the state into leaves and treedef."""
    if self._state is None:
      raise ValueError('Attempted to flatten a stale state.')
    treedef = (self._frozen,) + tuple(self._state.keys())
    leaves = tuple(self._state[k] for k in self._state.keys())
    return leaves, treedef

  @classmethod
  def tree_unflatten(cls, treedef: tuple, leaves: tuple):
    """Unflatten the state from leaves (values) and treedef (keys)."""
    self = object.__new__(cls)
    state = {}
    frozen, *keys = treedef
    values = leaves
    assert len(keys) == len(values)
    for key, value in zip(keys, values):
      state[key] = value
    self._state = state
    self._frozen = frozen
    return self


def initialize_state_index[PyTree](pytree: PyTree) -> PyTree:
  """Change all the values of `StateIndex.marker` from `object` to `str`, following the approach in
  `eqx.nn.make_with_state()`. This ensures that state constructed from different instances of
  the pytree are compatible, since rather than relying on `object` keys for lookup (which are
  different for each instance), we use `str` keys (which are consistent as long as the tree
  structure and traversal order is the same).
  """
  key_leaves, treedef = jax.tree.flatten_with_path(
    pytree, is_leaf=lambda x: isinstance(x, eqx.nn.StateIndex)
  )
  new_leaves = []
  for key, leaf in key_leaves:
    if isinstance(leaf, eqx.nn.StateIndex):
      leaf = eqx.nn.StateIndex(leaf.init)
      path_str = pytree.__class__.__name__ + jax.tree_util.keystr(key)
      object.__setattr__(leaf, 'marker', path_str)  # Circumvent frozen attribute error.
    new_leaves.append(leaf)
  pytree = jax.tree.unflatten(treedef, new_leaves)
  return pytree


# ==================================================================================================
# Entity

class BaseEnvState(eqx.Module):
  """Base class of an environment's state.

  Attributes:
    rng: Random key of the episode.
    steps: Number of steps taken in the episode.
    terminated: Whether the episode has ended (bool; sticky, see `Step`).
    local: Local states of the entities and transforms, keyed by their `StateIndex`.
  """
  rng: jax.Array
  steps: jax.Array
  terminated: jax.Array
  local: State
  def replace(self, **kwargs) -> T.Self:
    """Returns a copy with the given fields replaced."""
    return dataclasses.replace(self, **kwargs)

class BaseLocalState(eqx.Module):
  """Base class of an entity's or transform's own state, stored in `BaseEnvState.local`."""
  def replace(self, **kwargs) -> T.Self:
    """Returns a copy with the given fields replaced."""
    return dataclasses.replace(self, **kwargs)

Context = dict[str, T.Any]
"""Static information passed down the entity tree during `entity_initialize`. Each entity and
transform receives a shallow copy, so changes only reach its own subtree."""

class Transform[EnvState: BaseEnvState, LocalState: BaseLocalState | None](eqx.Module):
  """Modifies an entity's actions, observations and their spaces.

  An entity's transforms form a stack, listed innermost first in `Entity.transforms`:
  - `initialize` runs once for the environment, and `reset` at the start of each episode;
  - spaces are transformed from the innermost transform outward, for actions and observations alike;
  - action values are transformed from the outermost transform inward, ending at the entity;
  - observation values are transformed from the innermost transform outward, starting at the entity.

  Each method receives the space of its input. The defaults pass everything through unchanged.

  Attributes:
    index: Key of the transform's local state in `BaseEnvState.local`.
  """
  index: eqx.nn.StateIndex[LocalState]

  # `None` (stateless) needs a type-parameter default to typecheck, which requires Python 3.13.
  def __init__(self, local_state: LocalState = None):  # type: ignore
    """Initializes the transform with its initial local state (`None` if it has none)."""
    self.index = eqx.nn.StateIndex(local_state)

  def initialize(
    self,
    state: EnvState,
    context: Context,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
  ) -> EnvState:
    """Initialize static properties using the context passed by the parent entity tree."""
    return state

  def reset(self, state: EnvState) -> EnvState:
    """Reset the transform state at the beginning of each episode."""
    return state

  def action(
    self,
    space: ActionSpace,
    value: ArrayDict,
    state: EnvState,
  ) -> tuple[ArrayDict, EnvState]:
    """Transform action values. Override in subclasses that modify actions."""
    return value, state

  def action_space(self, space: ActionSpace, state: EnvState) -> ActionSpace:
    """Transform action space. Override in subclasses that modify action spaces."""
    return space

  def observation(
    self,
    space: ObservationSpace,
    value: ArrayDict,
    state: EnvState,
  ) -> tuple[ArrayDict, EnvState]:
    """Transform observation values. Override in subclasses that modify observations."""
    return value, state

  def observation_space(self, space: ObservationSpace, state: EnvState) -> ObservationSpace:
    """Transform observation space. Override in subclasses that modify observation spaces."""
    return space


class Entity[EnvState: BaseEnvState, LocalState: BaseLocalState | None](eqx.Module):
  """A node of the environment tree, such as a task, body or arena, with its own actions and
  observations.

  The `entity_*` functions traverse the tree. A child's action and observation keys appear in its
  parent's spaces prefixed with `<child name>/`, so `/` is reserved in keys and child names.

  Attributes:
    children: Child entities by name, in order.
    transforms: Transforms of this entity's own actions and observations, innermost first.
    index: Key of the entity's local state in `BaseEnvState.local`.
  """
  children: collections.OrderedDict[str, 'Entity[EnvState, T.Any]']
  transforms: list[Transform]  # Order is innermost to outermost.
  index: eqx.nn.StateIndex[LocalState]

  def __init__(
    self,
    children: T.Mapping[str, 'Entity[EnvState, T.Any] | None'] = types.MappingProxyType({}),
    transforms: T.Iterable[Transform | T.Literal[False] | None] = (),
    local_state: LocalState = None,  # type: ignore  # See `Transform.__init__`.
  ):
    """Initializes the entity.

    Args:
      children: Child entities by name. `None` entries are dropped, so optional children can be
        passed directly.
      transforms: Transforms, innermost first. `None` and `False` entries are dropped.
      local_state: Initial local state, or `None` if the entity has none.
    """
    self.children = collections.OrderedDict({k: v for k, v in children.items() if v is not None})
    self.transforms = [t for t in transforms if t]
    self.index = eqx.nn.StateIndex(local_state)

  def initialize(self, state: EnvState, context: Context) -> tuple[EnvState, Context]:
    """Initialize the entity. Override in subclasses that require initialization."""
    return state, context

  def action(self, state: EnvState, action: ArrayDict) -> EnvState:
    """Apply action to the state. Override in subclasses that support actions."""
    return state

  def action_space(self, state: EnvState) -> ActionSpace:
    """Return the action space. Override in subclasses that support actions."""
    return Dict()

  def observation(self, state: EnvState) -> tuple[ArrayDict, EnvState]:
    """Get observations from the state. Override in subclasses that support observations."""
    return collections.OrderedDict({}), state

  def observation_space(self, state: EnvState) -> ObservationSpace:
    """Return the observation space. Override in subclasses that support observations."""
    return Dict()


def entity_initialize[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
  context: Context,
) -> EnvState:
  """Initializes an entity, its transforms and its children, depth first.

  Args:
    entity: Root of the subtree to initialize.
    state: Environment state.
    context: Static information from the parent (see `Context`).

  Returns:
    The initialized state.

  Raises:
    KeyError: If a space key or child name contains `/`, or a child name equals a key of its
      parent's spaces.
  """
  # Shallow copying `context` is cheap and enables entity/transforms to modify it without worrying
  # about affecting the other subtrees.
  state, context = entity.initialize(state, context.copy())
  observation_space = entity.observation_space(state)
  action_space = entity.action_space(state)
  if any('/' in k for k in observation_space):
    raise KeyError(
      f"Reserved char '/' in original observation space: entity `{entity.__class__.__name__}`, "
      f'keys {list(observation_space.keys())}'
    )
  if any('/' in k for k in action_space):
    raise KeyError(
      f"Reserved char '/' in original action space: entity `{entity.__class__.__name__}`, "
      f'keys {list(action_space.keys())}'
    )

  for transform in entity.transforms:  # Transform spaces in original order.
    state = transform.initialize(state, context.copy(), observation_space, action_space)
    observation_space = transform.observation_space(observation_space, state)
    action_space = transform.action_space(action_space, state)
    if any('/' in k for k in observation_space):
      raise KeyError(
        f"Reserved char '/' in transformed observation space: entity "
        f'`{entity.__class__.__name__}`, transform `{transform.__class__.__name__}`, '
        f'keys {list(observation_space.keys())}'
      )
    if any('/' in k for k in action_space):
      raise KeyError(
        f"Reserved char '/' in transformed action space: entity "
        f'`{entity.__class__.__name__}`, transform `{transform.__class__.__name__}`, '
        f'keys {list(action_space.keys())}'
      )

  for name, child in entity.children.items():
    if '/' in name:
      raise KeyError(f"Reserved char '/' in child entity name: '{name}'")
    if name in observation_space:
      raise KeyError(f"Duplicate observation space key and child entity name: '{name}'")
    if name in action_space:
      raise KeyError(f"Duplicate action space key and child entity name: '{name}'")
    state = entity_initialize(child, state, context)

  return state


def entity_reset[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
) -> EnvState:
  """Resets the transforms of an entity and its children, depth first, at the start of an
  episode."""
  for transform in entity.transforms:
    state = transform.reset(state)
  for name, child in entity.children.items():
    state = entity_reset(child, state)
  return state


def entity_action[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
  action: Action,
) -> EnvState:
  """Applies an action to an entity and its children.

  The entity's own keys pass through its transforms from the outermost inward, then go to
  `Entity.action`. Keys prefixed with `<child name>/` go, without the prefix, to that child.

  Args:
    entity: Root of the subtree.
    state: Environment state.
    action: Action values of the subtree, keyed as in `entity_action_space`.

  Returns:
    The state after the action.

  Raises:
    KeyError: If one of the entity's own keys is missing.
    ValueError: If a value does not match its space, before or after a transform.
  """
  # Construct input action spaces for each transform. Need to do this first since spaces are
  # transformed in original order but action values are transformed in reverse order.
  spaces = [entity.action_space(state)]
  for transform in entity.transforms:  # Transform spaces in original order.
    spaces.append(transform.action_space(spaces[-1], state))

  # Validate, reorder, and separate self action from children actions.
  self_action: ArrayDict = collections.OrderedDict()
  for k in spaces[-1].keys():
    if k not in action:
      raise KeyError(
        f"Key missing in entity action: entity `{entity.__class__.__name__}`, expected key '{k}', "
        f'got action {action}'
      )
    self_action[k] = action[k]
  try:
    spaces[-1].validate(self_action)
  except Exception as e:
    raise ValueError(
      f'Validation error in entity action: entity `{entity.__class__.__name__}`'
    ) from e

  # Pass transformed self action to entity. Transform values in reverse order.
  for i, transform in reversed(list(enumerate(entity.transforms))):
    original_space = spaces[i]
    self_action, state = transform.action(original_space, self_action, state)
    try:
      original_space.validate(self_action)
    except Exception as e:
      raise ValueError(
        f'Validation error in transformed entity action: entity `{entity.__class__.__name__}`, '
        f'transform `{transform.__class__.__name__}`'
      ) from e
  state = entity.action(state, self_action)

  # Pass children actions to children.
  for name, entity in entity.children.items():
    child_action = {
      k.removeprefix(f'{name}/'): v for k, v in action.items() if k.startswith(f'{name}/')
    }
    state = entity_action(entity, state, T.cast(Action, child_action))

  return state


def entity_observation[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
) -> tuple[Observation, EnvState]:
  """Returns the observation of an entity and its children.

  The entity's own observation passes through its transforms from the innermost outward. Children's
  keys are prefixed with `<child name>/`.

  Args:
    entity: Root of the subtree.
    state: Environment state.

  Returns:
    The observation, keyed as in `entity_observation_space`, and the state.

  Raises:
    ValueError: If a value does not match its space, before or after a transform.
  """
  observation: Observation = collections.OrderedDict()

  # Get transformed self observation from entity.
  obs, state = entity.observation(state)
  space = entity.observation_space(state)
  try:
    space.validate(obs)
  except Exception as e:
    raise ValueError(
      f'Validation error in original entity observation: entity `{entity.__class__.__name__}`'
    ) from e
  for transform in entity.transforms:  # Transform values in original order.
    obs, state = transform.observation(space, obs, state)
    space = transform.observation_space(space, state)
    try:
      space.validate(obs)
    except Exception as e:
      raise ValueError(
        f'Validation error in transformed entity observation: entity '
        f'`{entity.__class__.__name__}`, transform `{transform.__class__.__name__}`'
      ) from e
  observation.update(obs)

  # Get children observations from children.
  for name, entity in entity.children.items():
    obs, state = entity_observation(entity, state)
    if len(obs) == 0: continue
    for k, v in obs.items():
      observation[f'{name}/{k}'] = v

  return observation, state


def entity_action_space[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
) -> ActionSpace:
  """Returns the transformed action space of an entity, with its children's keys prefixed by
  `<child name>/`."""
  action_space: ActionSpace = Dict()
  space = entity.action_space(state)
  for transform in entity.transforms:  # Transform spaces in original order.
    space = transform.action_space(space, state)
  action_space.update(space)
  for name, entity in entity.children.items():
    aspace = entity_action_space(entity, state)
    if len(aspace) == 0: continue
    for k, v in aspace.items():
      action_space[f'{name}/{k}'] = v
  return action_space


def entity_observation_space[EnvState: BaseEnvState](
  entity: Entity,
  state: EnvState,
) -> ObservationSpace:
  """Returns the transformed observation space of an entity, with its children's keys prefixed
  by `<child name>/`."""
  observation_space: ObservationSpace = Dict()
  space = entity.observation_space(state)
  for transform in entity.transforms:  # Transform spaces in original order.
    space = transform.observation_space(space, state)
  observation_space.update(space)
  for name, entity in entity.children.items():
    ospace = entity_observation_space(entity, state)
    if len(ospace) == 0: continue
    for k, v in ospace.items():
      observation_space[f'{name}/{k}'] = v
  return observation_space
