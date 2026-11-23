import collections
import collections.abc
import dataclasses
import re
import types
import typing as T

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import core, math, mjx_env

# ==================================================================================================
# Clipping


class ClipAction[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Clips action values to their `Box` space's bounds.

  Attributes:
    keys: The keys to clip, or `None` for all.
  """
  keys: list[str] | None = eqx.field(static=True)

  def __init__(self, keys: T.Iterable[str] | None = None):
    """Initializes the transform to clip `keys`, or all keys if `None`."""
    super().__init__()
    self.keys = list(keys) if keys is not None else None

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action with the selected `Box` values clipped to their bounds."""
    keys = self.keys if self.keys is not None else list(space.keys())
    act = collections.OrderedDict()
    for k, v in value.items():
      if k in keys:
        sp = space[k]
        if isinstance(sp, core.Box):
          v = jnp.clip(v, sp.low, sp.high)
      act[k] = v
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space unchanged."""
    return space


# ==================================================================================================
# Buffers


class BufferObservationSubstepsState(core.BaseLocalState):
  """Local state of `BufferObservationSubsteps`.

  Attributes:
    buffer: The observations of the last `sim_substeps` substeps per key, oldest first.
  """
  buffer: dict[str, jax.Array]


class BufferObservationSubsteps[EnvState: mjx_env.MjxEnvState](
  core.Transform[EnvState, BufferObservationSubstepsState]
):
  """Keeps the observations of the last env step's substeps in a buffer, and passes the newest
  through unchanged."""

  def __init__(self):
    """Initializes the transform with an empty buffer."""
    super().__init__(BufferObservationSubstepsState(buffer={}))

  @T.override
  def initialize(
    self,
    state: EnvState,
    context: core.Context,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
  ) -> EnvState:
    """Allocates a zero buffer of `sim_substeps` entries per observation key."""
    # Buffer is initialized by stacking zeros for each observation value. Order is oldest to newest.
    buffer_size = state.sim_substeps
    buffer = {
      k: jnp.zeros((buffer_size,) + sp.shape, dtype=sp.dtype)
      for k, sp in observation_space.items()
    }
    local = state.local.get(self.index)
    local = local.replace(buffer=buffer)
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def reset(self, state: EnvState) -> EnvState:
    """Zeros the buffer at the start of each episode."""
    local = state.local.get(self.index)
    buffer = {k: jnp.zeros_like(v) for k, v in local.buffer.items()}
    local = local.replace(buffer=buffer)
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Appends the observation to the buffer and returns it."""
    local = state.local.get(self.index)
    buffer = local.buffer.copy()

    # Every sim substep, shift the buffer and insert the new observation.
    for k in buffer.keys():
      assert k in value, f'{k} not in {list(value.keys())}'
      buffer[k] = jnp.roll(buffer[k], shift=-1, axis=0)
      buffer[k] = buffer[k].at[-1].set(value[k])
    local = local.replace(buffer=buffer)
    state = state.replace(local=state.local.set(self.index, local))

    # Get observation from the last index.
    obs = collections.OrderedDict({k: buffer[k][-1] for k in space.keys()})
    return obs, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space unchanged."""
    return space


class BufferActionSubstepsState(core.BaseLocalState):
  """Local state of `BufferActionSubsteps`.

  Attributes:
    buffer: The actions of the last env steps per key, oldest first.
    delay: Action delay: in seconds as configured, then per key in substeps after `initialize`.
    history: Number of past actions observed: as configured, then per key after `initialize`.
  """
  buffer: dict[str, jax.Array]
  delay: float | dict[str, float] = eqx.field(static=True)
  history: int | dict[str, int] = eqx.field(static=True)


class BufferActionSubsteps[EnvState: mjx_env.MjxEnvState](
  core.Transform[EnvState, BufferActionSubstepsState]
):
  """Delays actions by a fixed time, and optionally adds the last actions to the observation.

  Each env step's action is buffered at its first substep. Each substep applies the action that was
  current `delay` before it.

  Attributes:
    history_prefix: Prefix of the observation keys of the action history (`<prefix>_<action key>`).
  """
  history_prefix: str

  def __init__(
    self,
    delay: float | T.Mapping[str, float] = 0.,
    history: int | T.Mapping[str, int] = 0,
    history_prefix: str = 'action_history',
  ):
    """Initializes the transform.

    Args:
      delay: Action delay (s), for all keys or per key (0 for keys not listed). Rounded to substeps.
      history: Number of past env steps' actions to observe, for all keys or per key.
      history_prefix: Prefix of the observation keys of the action history.
    """
    super().__init__(BufferActionSubstepsState(
      buffer={},
      delay=float(delay) if isinstance(delay, (float, int)) else dict(delay),
      history=int(history) if isinstance(history, (float, int)) else dict(history),
    ))
    self.history_prefix = history_prefix

  def initialize(
    self,
    state: EnvState,
    context: core.Context,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
  ) -> EnvState:
    """Resolves the delay and history per action key, and allocates the buffers."""
    local = state.local.get(self.index)

    # Construct delay and history dicts using action space. Both values transformed into substeps.
    delay = local.delay
    if isinstance(delay, (float, int)):
      delay = {k: round(delay/state.sim_timestep) for k in action_space.keys()}
    else:
      delay = {k: round(delay.get(k, 0)/state.sim_timestep) for k in action_space.keys()}
    history = local.history
    if isinstance(history, (float, int)):
      history = {k: history for k in action_space.keys()}
    else:
      history = {k: history.get(k, 0) for k in action_space.keys()}

    # The buffer holds the current env step's action and, with delay, the `ceil(delay / substeps)`
    # previous ones that the delayed substeps read. With history, it holds at least the history.
    buffer_size = {
      k: max(1 + int(np.ceil(round(delay[k]/state.sim_substeps, 10))), history[k])
      for k in action_space.keys()
    }

    # Buffer is initialized by stacking zeros for each action value. Order is oldest to newest.
    buffer = {
      k: jnp.zeros((buffer_size[k],) + sp.shape, dtype=sp.dtype)
      for k, sp in action_space.items()
    }

    local = local.replace(buffer=buffer, delay=delay, history=history)
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def reset(self, state: EnvState) -> EnvState:
    """Zeros the buffer at the start of each episode."""
    local = state.local.get(self.index)
    buffer = {k: jnp.zeros_like(v) for k, v in local.buffer.items()}
    local = local.replace(buffer=buffer)
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Buffers the action at an env step's first substep, and returns the delayed action."""
    local = state.local.get(self.index)
    buffer = local.buffer.copy()

    # Every start of env step (not sim substep), shift the buffer and insert the new action. The
    # entire duration of the env step will thus show history with this last action.
    def update_buffer(buffer: dict[str, jax.Array]) -> dict[str, jax.Array]:
      """Shifts the buffer and appends the action."""
      for k in buffer.keys():
        assert k in value, f'{k} not in {list(value.keys())}'
        buffer[k] = jnp.roll(buffer[k], shift=-1, axis=0)
        buffer[k] = buffer[k].at[-1].set(value[k])
      return buffer

    def do_nothing(buffer: dict[str, jax.Array]) -> dict[str, jax.Array]:
      """Returns the buffer unchanged."""
      return buffer

    is_first_step = (state.substeps - 1) % state.sim_substeps == 0
    buffer = jax.lax.cond(is_first_step, update_buffer, do_nothing, buffer)
    local = local.replace(buffer=buffer)
    state = state.replace(local=state.local.set(self.index, local))

    # Get action from indices based on delay. If sim substeps in [1, sim_substeps] are normalized
    # within a single env step as t in [0, 1), then the actions of the env step are taken from
    # at most 2 adjacent buffer entries, with current index calculable using floor division.
    delay = T.cast(dict[str, int], local.delay)
    t = (state.substeps - 1) % state.sim_substeps
    idxs = {k: -1 + (t - v) // state.sim_substeps for k, v in delay.items()}
    act = collections.OrderedDict({k: buffer[k][idxs[k]] for k in space.keys()})

    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space unchanged."""
    return space

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the observation with the action history added."""
    local = state.local.get(self.index)
    history = T.cast(dict[str, int], local.history)
    buffer = local.buffer

    # Get observation from indices based on history.
    hist = {
      f'{self.history_prefix}_{k}': buffer[k][-h:]
      for k, h in history.items() if h > 0
    }
    obs = value.copy()
    obs.update(hist)
    return obs, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space with the action history's spaces added."""
    local = state.local.get(self.index)
    history = T.cast(dict[str, int], local.history)

    # Extend observation space with history entries.
    hist = {
      f'{self.history_prefix}_{k}': space[k].vectorize(h)
      for k, h in history.items() if h > 0
    }
    obs_space = space.copy()
    obs_space.update(hist)
    return obs_space


# ==================================================================================================
# Rescaling


class RescaleObservation[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Rescales `Box` observations piecewise-linearly from `[min, mid, max]` to `scale`, clipped."""
  min: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  mid: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  max: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  scale: tuple[float, float, float] | dict[str, tuple[float, float, float]] = eqx.field(static=True)
  signed: bool | dict[str, bool] = eqx.field(static=True)
  exclude: list[str] = eqx.field(static=True)

  def __init__(
    self,
    min: None | float | T.Mapping[str, math.ArrayLike] = None,
    mid: None | float | T.Mapping[str, math.ArrayLike] = None,
    max: None | float | T.Mapping[str, math.ArrayLike] = None,
    scale: T.Sequence[float] | T.Mapping[str, T.Sequence[float]] = (-1., 0., +1.),
    signed: bool | T.Mapping[str, bool] = True,
    exclude: T.Iterable[str] = (),
  ):
    """Initializes the transform.

    Args:
      min: Lower bound of the original values (all keys, or per key). Defaults to the space's.
      mid: Original value mapped to `scale`'s middle (all keys, or per key). Defaults to 0 if
        `signed`, otherwise the middle of the bounds; clipped into the bounds.
      max: Upper bound of the original values (all keys, or per key). Defaults to the space's.
      scale: `(low, high)` or `(low, mid, high)` of the rescaled values (all keys, or per key).
      signed: Whether `mid` defaults to 0 (all keys, or per key).
      exclude: Keys left unchanged.
    """
    super().__init__()
    assert (min is None) == (max is None), 'Must provide neither or both min and max'

    if min is None:
      self.min = {}
    elif isinstance(min, (float, int)):
      self.min = float(min)
    else:
      self.min = {k: math.float_tuple(v) for k, v in min.items()}

    if mid is None:
      self.mid = {}
    elif isinstance(mid, (float, int)):
      self.mid = float(mid)
    else:
      self.mid = {k: math.float_tuple(v) for k, v in mid.items()}

    if max is None:
      self.max = {}
    elif isinstance(max, (float, int)):
      self.max = float(max)
    else:
      self.max = {k: math.float_tuple(v) for k, v in max.items()}

    if isinstance(scale, collections.abc.Sequence):
      assert 2 <= len(scale) <= 3, f'Must provide (min, max) or (min, mid, max) scale: {scale}'
      self.scale = (
        float(scale[0]),
        float(scale[1]) if len(scale) == 3 else float((scale[0] + scale[-1])/2),
        float(scale[-1]),
      )
    else:
      self.scale = {
        k: (
          float(v[0]),
          float(v[1]) if len(v) == 3 else float((v[0] + v[-1])/2),
          float(v[-1]),
        )
        for k, v in scale.items()
      }

    if isinstance(signed, bool):
      self.signed = bool(signed)
    else:
      self.signed = {k: bool(v) for k, v in signed.items()}

    self.exclude = list(exclude)

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the rescaled observation."""
    obs = collections.OrderedDict()
    for k, v in value.items():
      sp = space[k]
      if k in self.exclude or not isinstance(sp, core.Box):
        # Don't transform excluded keys or non-Box spaces.
        obs[k] = v
        continue
      signed = self.signed.get(k, True) if isinstance(self.signed, dict) else self.signed
      scale = self.scale.get(k, (-1., 0., +1.)) if isinstance(self.scale, dict) else self.scale
      min = jnp.asarray(self.min.get(k, sp.low) if isinstance(self.min, dict) else self.min)
      max = jnp.asarray(self.max.get(k, sp.high) if isinstance(self.max, dict) else self.max)
      mid = jnp.asarray(
        self.mid.get(k, 0. if signed else (min + max)/2)
        if isinstance(self.mid, dict)
        else self.mid
      )
      mid = jnp.clip(mid, min, max)  # Ensure mid is within [min, max] for when 0 is outside range.
      y = math.rescale_linear3(v, min, mid, max, scale[0], scale[1], scale[2], clip=True)
      obs[k] = y
    return obs, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space with rescaled bounds."""
    obs_space = core.Dict()
    for k, sp in space.items():
      if k in self.exclude or not isinstance(sp, core.Box):
        # Don't transform excluded keys or non-Box spaces.
        obs_space[k] = sp
        continue
      signed = self.signed.get(k, True) if isinstance(self.signed, dict) else self.signed
      scale = self.scale.get(k, (-1., 0., +1.)) if isinstance(self.scale, dict) else self.scale
      min = jnp.asarray(self.min.get(k, sp.low) if isinstance(self.min, dict) else self.min)
      max = jnp.asarray(self.max.get(k, sp.high) if isinstance(self.max, dict) else self.max)
      mid = jnp.asarray(
        self.mid.get(k, 0. if signed else (min + max)/2)
        if isinstance(self.mid, dict)
        else self.mid
      )
      mid = jnp.clip(mid, min, max)  # Ensure mid is within [min, max] for when 0 is outside range.
      low = math.rescale_linear3(min, min, mid, max, scale[0], scale[1], scale[2], clip=True)
      high = math.rescale_linear3(max, min, mid, max, scale[0], scale[1], scale[2], clip=True)
      obs_space[k] = core.Box(
        low=jnp.broadcast_to(low, sp.shape).astype(sp.dtype),
        high=jnp.broadcast_to(high, sp.shape).astype(sp.dtype),
        shape=sp.shape,
        dtype=sp.dtype,
        name=sp.name,
        desc=sp.desc,
        labels=sp.labels,
      )
    return obs_space


class RescaleAction[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Rescales `Box` actions piecewise-linearly from `scale` to `[min, mid, max]`, clipped: the
  policy acts in `scale`."""
  min: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  mid: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  max: float | dict[str, tuple[float, ...]] = eqx.field(static=True)
  scale: tuple[float, float, float] | dict[str, tuple[float, float, float]] = eqx.field(static=True)
  signed: bool | dict[str, bool] = eqx.field(static=True)
  exclude: list[str] = eqx.field(static=True)

  def __init__(
    self,
    min: None | float | T.Mapping[str, math.ArrayLike] = None,
    mid: None | float | T.Mapping[str, math.ArrayLike] = None,
    max: None | float | T.Mapping[str, math.ArrayLike] = None,
    scale: T.Sequence[float] | T.Mapping[str, T.Sequence[float]] = (-1., 0., +1.),
    signed: bool | T.Mapping[str, bool] = True,
    exclude: T.Iterable[str] = (),
  ):
    """Initializes the transform.

    Args:
      min: Lower bound of the original values (all keys, or per key). Defaults to the space's.
      mid: Original value mapped to `scale`'s middle (all keys, or per key). Defaults to 0 if
        `signed`, otherwise the middle of the bounds; clipped into the bounds.
      max: Upper bound of the original values (all keys, or per key). Defaults to the space's.
      scale: `(low, high)` or `(low, mid, high)` of the rescaled values (all keys, or per key).
      signed: Whether `mid` defaults to 0 (all keys, or per key).
      exclude: Keys left unchanged.
    """
    super().__init__()
    assert (min is None) == (max is None), 'Must provide neither or both min and max'

    if min is None:
      self.min = {}
    elif isinstance(min, (float, int)):
      self.min = float(min)
    else:
      self.min = {k: math.float_tuple(v) for k, v in min.items()}

    if mid is None:
      self.mid = {}
    elif isinstance(mid, (float, int)):
      self.mid = float(mid)
    else:
      self.mid = {k: math.float_tuple(v) for k, v in mid.items()}

    if max is None:
      self.max = {}
    elif isinstance(max, (float, int)):
      self.max = float(max)
    else:
      self.max = {k: math.float_tuple(v) for k, v in max.items()}

    if isinstance(scale, collections.abc.Sequence):
      assert 2 <= len(scale) <= 3, f'Must provide (min, max) or (min, mid, max) scale: {scale}'
      self.scale = (
        float(scale[0]),
        float(scale[1]) if len(scale) == 3 else float((scale[0] + scale[-1])/2),
        float(scale[-1]),
      )
    else:
      self.scale = {
        k: (
          float(v[0]),
          float(v[1]) if len(v) == 3 else float((v[0] + v[-1])/2),
          float(v[-1]),
        )
        for k, v in scale.items()
      }

    if isinstance(signed, bool):
      self.signed = bool(signed)
    else:
      self.signed = {k: bool(v) for k, v in signed.items()}

    self.exclude = list(exclude)

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action rescaled to the original bounds."""
    act = collections.OrderedDict()
    for k, v in value.items():
      sp = space[k]
      if k in self.exclude or not isinstance(sp, core.Box):
        # Don't transform excluded keys or non-Box spaces.
        act[k] = v
        continue
      signed = self.signed.get(k, True) if isinstance(self.signed, dict) else self.signed
      scale = self.scale.get(k, (-1., 0., +1.)) if isinstance(self.scale, dict) else self.scale
      min = jnp.asarray(self.min.get(k, sp.low) if isinstance(self.min, dict) else self.min)
      max = jnp.asarray(self.max.get(k, sp.high) if isinstance(self.max, dict) else self.max)
      mid = jnp.asarray(
        self.mid.get(k, 0. if signed else (min + max)/2)
        if isinstance(self.mid, dict)
        else self.mid
      )
      mid = jnp.clip(mid, min, max)  # Ensure mid is within [min, max] for when 0 is outside range.
      y = math.rescale_linear3(v, scale[0], scale[1], scale[2], min, mid, max, clip=True)
      act[k] = y
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space with bounds in `scale`."""
    act_space = core.Dict()
    for k, sp in space.items():
      if k in self.exclude or not isinstance(sp, core.Box):
        # Don't transform excluded keys or non-Box spaces.
        act_space[k] = sp
        continue
      signed = self.signed.get(k, True) if isinstance(self.signed, dict) else self.signed
      scale = self.scale.get(k, (-1., 0., +1.)) if isinstance(self.scale, dict) else self.scale
      min = jnp.asarray(self.min.get(k, sp.low) if isinstance(self.min, dict) else self.min)
      max = jnp.asarray(self.max.get(k, sp.high) if isinstance(self.max, dict) else self.max)
      mid = jnp.asarray(
        self.mid.get(k, 0. if signed else (min + max)/2)
        if isinstance(self.mid, dict)
        else self.mid
      )
      mid = jnp.clip(mid, min, max)  # Ensure mid is within [min, max] for when 0 is outside range.
      low = math.rescale_linear3(min, min, mid, max, scale[0], scale[1], scale[2], clip=True)
      high = math.rescale_linear3(max, min, mid, max, scale[0], scale[1], scale[2], clip=True)
      act_space[k] = core.Box(
        low=jnp.broadcast_to(low, sp.shape).astype(sp.dtype),
        high=jnp.broadcast_to(high, sp.shape).astype(sp.dtype),
        shape=sp.shape,
        dtype=sp.dtype,
        name=sp.name,
        desc=sp.desc,
        labels=sp.labels,
      )
    return act_space


# ==================================================================================================
# Reducing, overriding and mapping


def extract_name_components(
  sp: core.Space,
) -> tuple[str, list[str] | None]:
  """From 'foo_bar:[a,b,c]', extract ('foo_bar', ['a', 'b', 'c'])."""
  spec = re.compile(r'^(?P<key>[^:]+):\[(?P<names>.+)\]$')
  match = spec.match(sp.name)
  if match is None:
    return sp.name, None
  key = match.group('key')
  names = match.group('names').split(',')
  return key, names


def reduce_name_components(
  sp: core.Space,
  uidxs: T.Sequence[int],
  replace_prefix: str | None = None,
) -> str:
  """Reduce 'foo_bar:[a,b,c]' to 'foo_bar:[a,c]' given indices [0, 2]."""
  key, names = extract_name_components(sp)
  if names is None or any(i < 0 or i >= len(names) for i in uidxs):
    return sp.name
  name = ','.join(names[i] for i in uidxs)
  name = f'{key if replace_prefix is None else replace_prefix}:[{name}]'
  return name


class ReduceAction[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Fixes some indices of 1-D `Box` actions to constants and removes them from the action space.

  A key whose indices are all fixed is removed from the action space.

  Attributes:
    cvalues: Constant values per key, at `cidxs`.
    cidxs: Constrained (fixed) indices per key.
    uidxs: Unconstrained indices per key, which remain in the action space.
  """
  cvalues: dict[str, tuple[float, ...]] = eqx.field(static=True)
  cidxs: dict[str, tuple[int, ...]] = eqx.field(static=True)
  uidxs: dict[str, tuple[int, ...]] = eqx.field(static=True)

  def __init__(
    self,
    constrain: T.Mapping[str, T.Sequence[float | None]] = types.MappingProxyType({}),
  ):
    """Initializes the transform with, per key, one entry per index: a constant to fix it to, or
    `None` to keep it in the action space."""
    super().__init__()
    self.cvalues = {}
    self.cidxs = {}
    self.uidxs = {}
    for k, v in constrain.items():
      if len(v) == 0: continue
      self.cvalues[k] = tuple(float(x) for x in v if x is not None)
      self.cidxs[k] = tuple(i for i, x in enumerate(v) if x is not None)
      self.uidxs[k] = tuple(i for i, x in enumerate(v) if x is None)

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the full action, with the constants inserted at their indices."""
    act = collections.OrderedDict()
    for k, sp in space.items():
      if k not in self.cvalues:
        # If no constraints, keep original.
        act[k] = value[k]
        continue

      # Construct an array of the original size, copying over constrained and unconstrained values.
      cvalues = jnp.asarray(self.cvalues[k])
      cidxs = jnp.asarray(self.cidxs[k], dtype=jnp.int32)
      uidxs = self.uidxs[k]  # Keep as tuple for length check.
      y = jnp.zeros(sp.shape, dtype=sp.dtype)
      y = y.at[cidxs].set(cvalues, indices_are_sorted=True, unique_indices=True)
      if len(uidxs) > 0:
        # Must check since value[k] does not exist when all indices are constrained, as the key is
        # removed entirely from the action space.
        uidxs = jnp.asarray(uidxs, dtype=jnp.int32)
        y = y.at[uidxs].set(value[k], indices_are_sorted=True, unique_indices=True)
      act[k] = y
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space without the fixed indices."""
    act_space = core.Dict()
    for k, sp in space.items():
      if k not in self.cvalues:
        # If no constraints, keep original.
        act_space[k] = sp
        continue

      uidxs = self.uidxs[k]
      if len(uidxs) == 0:
        # If all indices are constrained, remove the key entirely.
        continue
      if isinstance(sp, core.Box):
        assert len(sp.shape) == 1, f'Only 1D Box spaces are supported: {k} has shape {sp.shape}'

        name = reduce_name_components(sp, uidxs)
        labels = sp.labels and (
          [sp.labels[0]] if isinstance(sp.labels[0], str) else (
            [[sp.labels[0][i] for i in self.uidxs[k]]]
          )
        )
        uidxs = jnp.asarray(uidxs, dtype=jnp.int32)
        act_space[k] = core.Box(
          low=sp.low[uidxs],
          high=sp.high[uidxs],
          shape=(len(uidxs),),
          dtype=sp.dtype,
          name=name,
          desc=sp.desc,
          labels=labels,
        )
      else:
        raise NotImplementedError(f'Unsupported space type: {type(sp)}')
    return act_space


class OverrideAction[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Overrides action values at given indices with constants, without changing the action space.

  Attributes:
    ovalues: Constant values per key, at `oidxs`.
    oidxs: Overridden indices per key.
  """
  ovalues: dict[str, tuple[float, ...]] = eqx.field(static=True)
  oidxs: dict[str, tuple[int, ...]] = eqx.field(static=True)

  def __init__(
    self,
    override: T.Mapping[str, T.Sequence[float | None]] = types.MappingProxyType({}),
  ):
    """Initializes the transform with, per key, one entry per index: a constant to override it
    with, or `None` to leave it."""
    super().__init__()
    self.ovalues = {}
    self.oidxs = {}
    for k, v in override.items():
      oidxs = tuple(i for i, x in enumerate(v) if x is not None)
      if len(oidxs) == 0: continue
      self.ovalues[k] = tuple(float(x) for x in v if x is not None)
      self.oidxs[k] = oidxs

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action with the overridden indices set to their constants."""
    act = collections.OrderedDict()
    for k, v in value.items():
      if k not in self.ovalues:
        # If no overrides, keep original.
        act[k] = v
        continue

      # Overwrite only the specified indices; leave the rest of the value untouched.
      ovalues = jnp.asarray(self.ovalues[k])
      oidxs = jnp.asarray(self.oidxs[k], dtype=jnp.int32)
      act[k] = v.at[oidxs].set(ovalues, indices_are_sorted=True, unique_indices=True)
    return act, state


class MapAction[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Splits 1-D `Box` actions into new keys, each taking the indices selected by a mask.

  The masks of each original key must cover all its indices, without overlap. The new keys replace
  the original ones in the action space; other keys pass through.

  Attributes:
    remap: Per new key, `(original key, mask)`: which of the original key's indices it takes.
    map_sizes: Size of each original key.
  """
  remap: dict[str, tuple[str, tuple[bool, ...]]] = eqx.field(static=True)
  map_sizes: dict[str, int] = eqx.field(static=True)

  def __init__(
    self,
    remap: T.Mapping[str, tuple[str, T.Sequence[bool]]],
  ):
    """Initializes the transform with, per new key, `(original key, mask)`.

    Raises:
      AssertionError: If an original key's masks differ in size, overlap, or leave an index
        unmapped.
    """
    super().__init__()
    self.remap = dict({new_k: (k, tuple(v)) for new_k, (k, v) in remap.items()})
    self.map_sizes = {}
    overlap = {}
    for _, (k, mask) in remap.items():
      if k in self.map_sizes:
        assert self.map_sizes[k] == len(mask), (
          f'{k} must have consistent mask size: {self.map_sizes[k]} vs {len(mask)}'
        )
        assert not any([reserved and requested for reserved, requested in zip(overlap[k], mask)]), (
          f'{k} has overlapping remap entries: {overlap[k]} vs {mask}'
        )
      self.map_sizes[k] = len(mask)
      overlap[k] = tuple(a or b for a, b in zip(overlap[k], mask)) if k in overlap else mask
    for k, mask in overlap.items():
      assert all(mask), f'{k} has unmapped indices: {mask}'

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action with the new keys' values gathered back into their original keys."""
    new_keys = set(self.remap)
    act = collections.OrderedDict({k: v for k, v in value.items() if k not in new_keys})
    for k, size in self.map_sizes.items():
      if k in space:
        act[k] = jnp.zeros(size, dtype=space[k].dtype)
    for new_k, (k, mask) in self.remap.items():
      if k not in space:
        continue
      indices = jnp.asarray([i for i, m in enumerate(mask) if m], dtype=jnp.int32)
      act[k] = act[k].at[indices].set(value[new_k])
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space with the original keys replaced by the new keys."""
    act_space = core.Dict({k: sp for k, sp in space.items() if k not in self.map_sizes})
    for new_k, (k, mask) in self.remap.items():
      if k not in space:
        continue
      sp = space[k]
      assert isinstance(sp, core.Box) and len(sp.shape) == 1, 'Only 1D Box spaces are supported'
      uidxs = tuple(i for i, m in enumerate(mask) if m)
      indices = jnp.asarray(uidxs, dtype=jnp.int32)
      labels = sp.labels and (
        [sp.labels[0]] if isinstance(sp.labels[0], str) else [[sp.labels[0][i] for i in uidxs]]
      )
      act_space[new_k] = core.Box(
        low=sp.low[indices],
        high=sp.high[indices],
        shape=(len(uidxs),),
        dtype=sp.dtype,
        name=reduce_name_components(sp, uidxs, replace_prefix=new_k),
        desc=sp.desc,
        labels=labels,
      )
    return act_space


class ReduceObservation[EnvState: core.BaseEnvState](core.Transform[EnvState, None]):
  """Keeps or removes observation keys, or selects and reorders the indices of 1-D `Box` ones.

  Attributes:
    idxs: Per key, whether to keep it (`bool`), or the indices to keep, in order.
  """
  idxs: dict[str, bool | tuple[int, ...]] = eqx.field(static=True)

  def __init__(
    self,
    filter: T.Mapping[str, bool | T.Sequence[int]] = types.MappingProxyType({}),
  ):
    """Initializes the transform with, per key, whether to keep it or the indices to keep (an empty
    sequence removes it). Keys not listed are kept."""
    super().__init__()
    self.idxs = {}
    for k, v in filter.items():
      if isinstance(v, bool):
        self.idxs[k] = v
      else:
        if len(v) == 0:
          # If no specified indices, remove the key entirely.
          self.idxs[k] = False
        else:
          # Since specified indices are not sorted, they can both filter and reorder values.
          self.idxs[k] = math.int_tuple(v)

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the observation with the selected keys and indices."""
    obs = collections.OrderedDict()
    for k, v in value.items():
      sp = space[k]
      if k not in self.idxs:
        # If no filter, keep original.
        obs[k] = v
        continue

      idxs = self.idxs[k]
      if isinstance(idxs, bool):
        # If filter is bool, keep or remove the key entirely.
        if idxs:
          obs[k] = v
      else:
        # If filter is tuple, keep only values at the specified indices.
        idxs = jnp.asarray(idxs, dtype=jnp.int32)
        if isinstance(sp, core.Box):
          assert len(sp.shape) == 1, f'Only 1D Box spaces are supported: {k} has shape {sp.shape}'
          obs[k] = v[idxs]
        else:
          raise NotImplementedError(f'Unsupported space type: {type(sp)}')
    return obs, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space with the selected keys and indices."""
    obs_space = core.Dict()
    for k, sp in space.items():
      if k not in self.idxs:
        # If no filter, keep original.
        obs_space[k] = sp
        continue

      idxs = self.idxs[k]
      if isinstance(idxs, bool):
        # If filter is bool, keep or remove the key entirely.
        if idxs:
          obs_space[k] = sp
      else:
        # If filter is array, keep only values at the specified indices.
        idxs_arr = jnp.asarray(idxs, dtype=jnp.int32)
        if isinstance(sp, core.Box):
          assert len(sp.shape) == 1, f'Only 1D Box spaces are supported: {k} has shape {sp.shape}'
          labels = sp.labels and (
            [sp.labels[0]] if isinstance(sp.labels[0], str) else (
              [[sp.labels[0][i] for i in idxs]]
            )
          )
          obs_space[k] = core.Box(
            low=sp.low[idxs_arr],
            high=sp.high[idxs_arr],
            shape=(len(idxs_arr),),
            dtype=sp.dtype,
            name=reduce_name_components(sp, idxs),
            desc=sp.desc,
            labels=labels,
          )
        else:
          raise NotImplementedError(f'Unsupported space type: {type(sp)}')
    return obs_space


def _transform_box(
  space: core.Box,
  name: str = '',
  desc: str = '',
  name_prefix: str = '',
  desc_prefix: str = '',
  low: jax.Array | None = None,
  high: jax.Array | None = None,
  labels: T.Sequence[str | T.Sequence[str]] | None = None,
) -> core.Box:
  """Returns `space` with new bounds or metadata.

  Args:
    space: The space.
    name: New name. Takes precedence over `name_prefix`.
    desc: New description. Takes precedence over `desc_prefix`.
    name_prefix: New name prefix, replacing the part before `:` (`foo:[...]` -> `bar:[...]`).
    desc_prefix: Prefix of the description (`World` -> `Hello world`).
    low: New lower bounds.
    high: New upper bounds.
    labels: New labels.
  """
  if name:
    pass  # Use the new name.
  elif name_prefix:
    # Replace name prefix "foo:..." -> "bar:...".
    name = re.sub(r'^(?P<name>[^:]+):', f'{name_prefix}:', space.name)
  else:
    name = space.name

  if desc:
    pass  # Use the new description.
  elif desc_prefix:
    # Prepend description prefix "World" -> "Hello world" (lowercased first).
    desc = space.desc
    desc = desc[0].lower() + desc[1:] if desc else ''
    desc = f'{desc_prefix} {desc}' if desc else desc_prefix
  else:
    desc = space.desc

  if low is None:
    low = space.low
  if high is None:
    high = space.high

  if labels is None:
    labels = space.labels

  return dataclasses.replace(space, low=low, high=high, name=name, desc=desc, labels=labels)


# ==================================================================================================
# Controllers


class ProportionalDerivativeControllerState(core.BaseLocalState):
  """Local state of `ProportionalDerivativeController`.

  Attributes:
    trq_space: Space of the feedforward torque action.
    pos_space: Space of the target position action (the joint range, widened by `pos_margin`).
    vel_space: Space of the target velocity action.
    pos: The last observed positions.
    vel: The last observed velocities.
  """
  trq_space: core.Box
  pos_space: core.Box
  vel_space: core.Box
  pos: jax.Array
  vel: jax.Array

class ProportionalDerivativeController[EnvState: core.BaseEnvState](
  core.Transform[EnvState, ProportionalDerivativeControllerState]
):
  """A proportional-derivative (PD) controller: turns target positions and velocities into torques.

  The torque is `trq + kp * (pos_d - pos) + kd * (vel_d - vel) + kp_passive * (pos_default - pos)`,
  clipped to the torque space, where each term is present only if configured. `pos` and `vel` are
  the last observed values. The feedforward `trq`, target `pos_d` and `vel_d`, and gains `kp` and
  `kd` are actions if their `actionable_*` key is set, and fixed values otherwise.
  """
  actionable_trq: str | None = eqx.field(static=True)
  actionable_pos: str | None = eqx.field(static=True)
  actionable_vel: str | None = eqx.field(static=True)
  actionable_kp: str | None = eqx.field(static=True)
  actionable_kd: str | None = eqx.field(static=True)
  kp: tuple[float, ...] | None = eqx.field(static=True)
  kd: tuple[float, ...] | None = eqx.field(static=True)
  kp_passive: tuple[float, ...] | None = eqx.field(static=True)
  pos_default: tuple[float, ...] = eqx.field(static=True)
  vel_default: tuple[float, ...] = eqx.field(static=True)
  pos_margin: tuple[float, ...] = eqx.field(static=True)
  obskey_pos: str = eqx.field(static=True)
  obskey_vel: str = eqx.field(static=True)
  actkey_trq: str = eqx.field(static=True)
  additive_trq: bool = eqx.field(static=True)

  def __init__(
    self,
    actionable_trq: str | None = None,
    actionable_pos: str | None = None,
    actionable_vel: str | None = None,
    actionable_kp: str | None = None,
    actionable_kd: str | None = None,
    kp: math.ArrayLike | None = None,
    kd: math.ArrayLike | None = None,
    kp_passive: math.ArrayLike | None = None,
    pos_default: math.ArrayLike = 0.,
    vel_default: math.ArrayLike = 0.,
    pos_margin: math.ArrayLike = 0.,
    obskey_pos: str = 'joints_pos',
    obskey_vel: str = 'joints_vel',
    actkey_trq: str = 'joints_trq',
    additive_trq: bool = False,
    undamped: bool = False,
  ):
    """Initializes the controller.

    Args:
      actionable_trq: Action key of a feedforward torque, or `None` for none.
      actionable_pos: Action key of the target positions, or `None` to use `pos_default`.
      actionable_vel: Action key of the target velocities, or `None` to use `vel_default`.
      actionable_kp: Action key of the position gains, or `None` to use `kp`.
      actionable_kd: Action key of the velocity gains, or `None` to use `kd`.
      kp: Position gains; with `actionable_kp`, their upper bound (default 100).
      kd: Velocity gains (with `actionable_kd`, the upper bound is 10).
      kp_passive: Gains of a passive spring toward `pos_default`, or `None` for none.
      pos_default: Target positions without `actionable_pos`, and the passive spring's rest pose.
      vel_default: Target velocities without `actionable_vel`.
      pos_margin: Widening of the target position space beyond the joint range, as a fraction of it.
      obskey_pos: Observation key of the positions.
      obskey_vel: Observation key of the velocities.
      actkey_trq: Action key of the output torques.
      additive_trq: Whether to add the torque to the incoming `actkey_trq` action (which then stays
        in the action space) instead of replacing it.
      undamped: Whether no velocity term is intended (permits neither `kd` nor `actionable_kd`).
    """
    super().__init__(
      ProportionalDerivativeControllerState(
        trq_space=core.Box(0, 0),
        pos_space=core.Box(0, 0),
        vel_space=core.Box(0, 0),
        pos=jnp.zeros(()),
        vel=jnp.zeros(()),
      )
    )
    # Configure action space.
    self.actionable_trq = actionable_trq
    self.actionable_pos = actionable_pos
    self.actionable_vel = actionable_vel
    self.actionable_kp = actionable_kp
    self.actionable_kd = actionable_kd
    # If kp is provided alone, it is used as kp. If both are provided, kp is the maximum actionable
    # kp. If actionable_kp is provided alone, the maximum actionable kp is 100.
    assert actionable_kp or kp is not None
    assert undamped or actionable_kd or kd is not None
    self.kp = math.float_tuple(kp) if kp is not None else None
    self.kd = math.float_tuple(kd) if kd is not None else None
    self.kp_passive = math.float_tuple(kp_passive) if kp_passive is not None else None
    self.pos_default = math.float_tuple(pos_default)
    self.vel_default = math.float_tuple(vel_default)
    self.pos_margin = math.float_tuple(pos_margin)

    # Store keys for observation and action.
    self.obskey_pos = obskey_pos
    self.obskey_vel = obskey_vel
    self.actkey_trq = actkey_trq
    self.additive_trq = additive_trq

  @T.override
  def initialize(
    self,
    state: EnvState,
    context: core.Context,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
  ) -> EnvState:
    """Derives the controller's action spaces from the torque and joint spaces."""
    # Initialize arrays with correct shape to avoid jit recompilation. Cache limits for Box spaces.
    local = state.local.get(self.index)

    trq_sp = action_space[self.actkey_trq]
    assert isinstance(trq_sp, core.Box), 'Must have Box space for trq action'
    pos_sp = observation_space[self.obskey_pos]
    vel_sp = observation_space[self.obskey_vel]
    assert isinstance(pos_sp, core.Box), 'Must have Box space for pos observation'
    assert isinstance(vel_sp, core.Box), 'Must have Box space for vel observation'
    assert pos_sp.shape == vel_sp.shape, 'Must have same shape of pos and vel observations'
    assert pos_sp.shape == trq_sp.shape, 'Must have same shape of pos observation and trq action'

    # Compute position action space margins as a fraction of the joint/tendon range.
    pos_margin = (pos_sp.high - pos_sp.low) * jnp.array(self.pos_margin)

    local = local.replace(
      trq_space=_transform_box(
        trq_sp,
        desc_prefix='Feedforward extra',
      ),
      pos_space=_transform_box(
        pos_sp,
        desc_prefix='Feedback target',
        low=pos_sp.low - pos_margin,
        high=pos_sp.high + pos_margin,
      ),
      vel_space=_transform_box(
        vel_sp,
        desc_prefix='Feedback target',
      ),
      pos=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
      vel=jnp.zeros(vel_sp.shape, dtype=vel_sp.dtype),
    )
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def reset(self, state: EnvState) -> EnvState:
    """Zeros the stored position and velocity at the start of each episode."""
    local = state.local.get(self.index)
    local = local.replace(pos=jnp.zeros_like(local.pos), vel=jnp.zeros_like(local.vel))
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Stores the observed position and velocity, and returns the observation unchanged."""
    local = state.local.get(self.index)
    local = local.replace(pos=value[self.obskey_pos], vel=value[self.obskey_vel])
    state = state.replace(local=state.local.set(self.index, local))
    return value, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space unchanged."""
    return space

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action with the controller's keys replaced by the torque."""
    local = state.local.get(self.index)

    # Initialize torques and add feedforward trq from action.
    trq_sp = T.cast(core.Box, space[self.actkey_trq])
    trq = jnp.zeros(trq_sp.shape, dtype=trq_sp.dtype)
    if self.actionable_trq:
      trq = trq + value[self.actionable_trq]

    # Get desired target pos/vel from action or fixed values.
    pos_d = value[self.actionable_pos] if self.actionable_pos else jnp.asarray(self.pos_default)
    vel_d = value[self.actionable_vel] if self.actionable_vel else jnp.asarray(self.vel_default)
    pos_d_passive = jnp.asarray(self.pos_default)

    # Get kp/kd from action or fixed values. Terms are excluded if not provided in either.
    if self.actionable_kp:
      kp = value[self.actionable_kp]
      trq = trq + kp * (pos_d - local.pos)
    elif self.kp is not None:
      kp = jnp.asarray(self.kp)
      trq = trq + kp * (pos_d - local.pos)

    if self.actionable_kd:
      kd = value[self.actionable_kd]
      trq = trq + kd * (vel_d - local.vel)
    elif self.kd is not None:
      kd = jnp.asarray(self.kd)
      trq = trq + kd * (vel_d - local.vel)

    if self.kp_passive is not None:
      passive_kp = jnp.asarray(self.kp_passive)
      trq = trq + passive_kp * (pos_d_passive - local.pos)

    # Clip torques to the action space limits.
    trq = jnp.clip(trq, trq_sp.low, trq_sp.high)

    # Transform the action to the original space, removing controller-specific keys.
    act = value.copy()
    if self.actionable_trq: del act[self.actionable_trq]
    if self.actionable_pos: del act[self.actionable_pos]
    if self.actionable_vel: del act[self.actionable_vel]
    if self.actionable_kp: del act[self.actionable_kp]
    if self.actionable_kd: del act[self.actionable_kd]
    if self.additive_trq:
      act[self.actkey_trq] += trq
    else:
      act[self.actkey_trq] = trq
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space with the torque replaced by the controller's actions."""
    local = state.local.get(self.index)
    act_space = space.copy()
    if not self.additive_trq:
      del act_space[self.actkey_trq]
    if self.actionable_trq:
      act_space[self.actionable_trq] = local.trq_space
    if self.actionable_pos:
      act_space[self.actionable_pos] = local.pos_space
    if self.actionable_vel:
      act_space[self.actionable_vel] = local.vel_space
    if self.actionable_kp:
      act_space[self.actionable_kp] = _transform_box(
        local.pos_space,
        name=self.actionable_kp,
        desc_prefix='Proportional gain (trq/pos) for',
        low=jnp.zeros(local.pos_space.shape, dtype=local.pos_space.dtype),
        high=(
          jnp.ones(local.pos_space.shape, dtype=local.pos_space.dtype)
          * jnp.asarray((100 if self.kp is None else self.kp))
        )
      )
    if self.actionable_kd:
      act_space[self.actionable_kd] = _transform_box(
        local.vel_space,
        name=self.actionable_kd,
        desc_prefix='Derivative gain (trq/vel) for',
        low=jnp.zeros(local.vel_space.shape, dtype=local.vel_space.dtype),
        high=jnp.ones(local.vel_space.shape, dtype=local.vel_space.dtype) * 10,
      )
    return act_space


class SpringMuscleControllerState(core.BaseLocalState):
  """Local state of `SpringMuscleController`.

  Attributes:
    flx_space: Space of the flexor activation action, [0, 1].
    ext_space: Space of the extensor activation action, [0, 1].
    act_flx: Filtered flexor activations.
    act_ext: Filtered extensor activations.
    pos_space: Space of the joint positions (the joint range).
    pos: The last observed positions.
    vel: The last observed velocities.
    kf: Flexor stiffness.
    ke: Extensor stiffness.
    cf: Flexor offset.
    ce: Extensor offset.
  """
  # Control spaces
  flx_space: core.Box
  ext_space: core.Box
  # Filtered muscle activations
  act_flx: jax.Array
  act_ext: jax.Array
  # Observation spaces and buffers
  pos_space: core.Box
  pos: jax.Array
  vel: jax.Array
  # Spring muscle model parameters
  kf: jax.Array
  ke: jax.Array
  cf: jax.Array
  ce: jax.Array

class SpringMuscleController[EnvState: core.BaseEnvState](
  core.Transform[EnvState, SpringMuscleControllerState]
):
  """A spring-muscle controller: turns flexor and extensor activations in [0, 1] into torques.

  Each joint has a flexor and an extensor spring. With joint range `[low, high]`, position `q` and
  activations `flx` and `ext`, the torque is

    trq = ext * ke * (high - q + ce) - flx * kf * (q - low + cf)
          - kp_passive * (q - poseq) - kd * dq,

  clipped to the torque space: the extensor pushes toward `high` and the flexor toward `low`. The
  stiffnesses `kf`, `ke` and offsets `cf`, `ce` are set so that, at full co-contraction
  (`flx = ext = 1`), the joint rests at `poseq` with stiffness `keq`, and so that the flexor alone
  gives `trq_max_flx` at `high` and the extensor alone `trq_max_ext` at `low`:

    kf = (keq * (poseq - low) + trq_max_flx - trq_max_ext) / (high - low)
    ke = (keq * (high - poseq) + trq_max_ext - trq_max_flx) / (high - low)
    cf = trq_max_flx / kf - (high - low)
    ce = trq_max_ext / ke - (high - low)
  """
  # Convenient reparameterization of the muscle model (kf, ke, cf, ce).
  poseq: tuple[float, ...] = eqx.field(static=True)  # Equilibrium pose for flx=ext=1.
  keq: tuple[float, ...] = eqx.field(static=True)  # Stiffness at equilibrium for flx=ext=1.
  trq_max_flx: tuple[float, ...] = eqx.field(static=True)  # Max torque from flexor at flx=1, ext=0.
  trq_max_ext: tuple[float, ...] = eqx.field(static=True)

  # Passive torques.
  kp_passive: tuple[float, ...] | None = eqx.field(static=True)
  kd: tuple[float, ...] | None = eqx.field(static=True)

  # Passive observation keys
  obskey_pos: str = eqx.field(static=True)
  obskey_vel: str = eqx.field(static=True)

  # Muscle action input keys
  actkey_flx: str = eqx.field(static=True)
  actkey_ext: str = eqx.field(static=True)

  # Torque action output key
  actkey_trq: str = eqx.field(static=True)
  act_time: tuple[float, ...] | None = eqx.field(static=True)
  additive_trq: bool = eqx.field(static=True)

  def __init__(
    self,
    poseq: math.ArrayLike,
    keq: math.ArrayLike,
    trq_max_flx: math.ArrayLike,
    trq_max_ext: math.ArrayLike,
    kp_passive: math.ArrayLike | None = None,
    kd: math.ArrayLike | None = None,
    act_time: math.ArrayLike | None = None,
    obskey_pos: str = 'joints_pos',
    obskey_vel: str = 'joints_vel',
    actkey_flx: str = 'muscle_flx',
    actkey_ext: str = 'muscle_ext',
    actkey_trq: str = 'joints_trq',
    additive_trq: bool = False,
  ):
    """Initializes the controller.

    Args:
      poseq: Equilibrium pose for flx=ext=1 and passive stiffness target.
      keq: Effective stiffness at equilibrium for flx=ext=1.
      trq_max_flx: Max possible torque from flexor at flx=1, ext=0.
      trq_max_ext: Max possible torque from extensor at flx=0, ext=1.
      kp_passive: Proportional gain for passive torque.
      kd: Derivative gain for passive torque.
      act_time: Optional low-pass filter time constant for muscle activations.
      obskey_pos: Observation key for joint positions.
      obskey_vel: Observation key for joint velocities.
      actkey_flx: Action key for flexor activation.
      actkey_ext: Action key for extensor activation.
      actkey_trq: Action key for joint torques.
      additive_trq: Whether to add the torque to the incoming `actkey_trq` action (which then stays
        in the action space) instead of replacing it.
    """
    super().__init__(
      SpringMuscleControllerState(
        flx_space=core.Box(0, 0),
        ext_space=core.Box(0, 0),
        act_flx=jnp.zeros(()),
        act_ext=jnp.zeros(()),
        pos_space=core.Box(0, 0),
        pos=jnp.zeros(()),
        vel=jnp.zeros(()),
        kf=jnp.zeros(()),
        ke=jnp.zeros(()),
        cf=jnp.zeros(()),
        ce=jnp.zeros(()),
      )
    )
    self.poseq = math.float_tuple(poseq)
    self.keq = math.float_tuple(keq)
    self.trq_max_flx = math.float_tuple(trq_max_flx)
    self.trq_max_ext = math.float_tuple(trq_max_ext)
    self.kp_passive = math.float_tuple(kp_passive) if kp_passive is not None else (0.,)
    self.kd = math.float_tuple(kd) if kd is not None else (0.,)
    self.act_time = math.float_tuple(act_time) if act_time is not None else None
    self.obskey_pos = obskey_pos
    self.obskey_vel = obskey_vel
    self.actkey_flx = actkey_flx
    self.actkey_ext = actkey_ext
    self.actkey_trq = actkey_trq
    self.additive_trq = additive_trq

  @T.override
  def initialize(
    self,
    state: EnvState,
    context: core.Context,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
  ) -> EnvState:
    """Derives the controller's action spaces from the torque and joint spaces."""
    # Initialize arrays with correct shape to avoid jit recompilation. Cache limits for Box spaces.
    local = state.local.get(self.index)

    trq_sp = action_space[self.actkey_trq]
    assert isinstance(trq_sp, core.Box), 'Must have Box space for trq action'
    pos_sp = observation_space[self.obskey_pos]
    vel_sp = observation_space[self.obskey_vel]
    assert isinstance(pos_sp, core.Box), 'Must have Box space for pos observation'
    assert isinstance(vel_sp, core.Box), 'Must have Box space for vel observation'
    assert pos_sp.shape == vel_sp.shape, 'Must have same shape of pos and vel observations'
    assert pos_sp.shape == trq_sp.shape, 'Must have same shape of pos observation and trq action'

    poseq = jnp.asarray(self.poseq)
    keq = jnp.asarray(self.keq)
    trq_max_flx = jnp.asarray(self.trq_max_flx)
    trq_max_ext = jnp.asarray(self.trq_max_ext)

    # Reparameterization of muscle model parameters.
    kf = (keq * (poseq - pos_sp.low) + (trq_max_flx - trq_max_ext)) / (pos_sp.high - pos_sp.low)
    ke = (keq * (pos_sp.high - poseq) + (trq_max_ext - trq_max_flx)) / (pos_sp.high - pos_sp.low)
    cf = trq_max_flx / kf - (pos_sp.high - pos_sp.low)
    ce = trq_max_ext / ke - (pos_sp.high - pos_sp.low)

    local = local.replace(
      flx_space=_transform_box(
        pos_sp,
        name_prefix=self.actkey_flx,
        desc_prefix='Flexor activation',
        low=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
        high=jnp.ones(pos_sp.shape, dtype=pos_sp.dtype),
      ),
      ext_space=_transform_box(
        pos_sp,
        name_prefix=self.actkey_ext,
        desc_prefix='Extensor activation',
        low=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
        high=jnp.ones(pos_sp.shape, dtype=pos_sp.dtype),
      ),
      act_flx=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
      act_ext=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
      pos_space=pos_sp,
      pos=jnp.zeros(pos_sp.shape, dtype=pos_sp.dtype),
      vel=jnp.zeros(vel_sp.shape, dtype=vel_sp.dtype),
      kf=kf,
      ke=ke,
      cf=cf,
      ce=ce,
    )
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def reset(self, state: EnvState) -> EnvState:
    """Zeros the filtered activations and stored position and velocity at each episode's start."""
    local = state.local.get(self.index)
    local = local.replace(
      act_flx=jnp.zeros_like(local.act_flx),
      act_ext=jnp.zeros_like(local.act_ext),
      pos=jnp.zeros_like(local.pos),
      vel=jnp.zeros_like(local.vel),
    )
    state = state.replace(local=state.local.set(self.index, local))
    return state

  @T.override
  def observation(
    self,
    space: core.ObservationSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Stores the observed position and velocity, and returns the observation unchanged."""
    local = state.local.get(self.index)
    local = local.replace(pos=value[self.obskey_pos], vel=value[self.obskey_vel])
    state = state.replace(local=state.local.set(self.index, local))
    return value, state

  @T.override
  def observation_space(
    self,
    space: core.ObservationSpace,
    state: EnvState,
  ) -> core.ObservationSpace:
    """Returns the observation space unchanged."""
    return space

  @T.override
  def action(
    self,
    space: core.ActionSpace,
    value: core.ArrayDict,
    state: EnvState,
  ) -> tuple[core.ArrayDict, EnvState]:
    """Returns the action with the activations replaced by the torque."""
    local = state.local.get(self.index)

    flx = jnp.clip(value[self.actkey_flx], 0.0, 1.0)
    ext = jnp.clip(value[self.actkey_ext], 0.0, 1.0)
    if self.act_time is None:
      act_flx = flx
      act_ext = ext
    else:
      act_time = jnp.asarray(self.act_time)
      tau = 4 * state.sim_timestep / act_time
      act_flx = local.act_flx + tau * (flx - local.act_flx)
      act_ext = local.act_ext + tau * (ext - local.act_ext)

    local = local.replace(act_flx=act_flx, act_ext=act_ext)
    state = state.replace(local=state.local.set(self.index, local))

    trq_sp = T.cast(core.Box, space[self.actkey_trq])
    trq = jnp.zeros(trq_sp.shape, dtype=trq_sp.dtype)

    # Spring muscle model torque calculation.
    xf = local.pos - local.pos_space.low + local.cf
    xe = local.pos_space.high - local.pos + local.ce
    trqf = act_flx * local.kf * xf
    trqe = act_ext * local.ke * xe

    trq = trq + trqe - trqf

    # Passive torque calculations.
    poseq = jnp.asarray(self.poseq)
    kp_passive = jnp.asarray(self.kp_passive)
    kd = jnp.asarray(self.kd)
    trq = trq - kp_passive * (local.pos - poseq)
    trq = trq - kd * local.vel

    # Clip torques to the action space limits.
    trq = jnp.clip(trq, trq_sp.low, trq_sp.high)

    # Transform the action to the original space, removing controller-specific keys.
    act = value.copy()
    del act[self.actkey_flx]
    del act[self.actkey_ext]
    if self.additive_trq:
      act[self.actkey_trq] += trq
    else:
      act[self.actkey_trq] = trq
    return act, state

  @T.override
  def action_space(
    self,
    space: core.ActionSpace,
    state: EnvState,
  ) -> core.ActionSpace:
    """Returns the action space with the torque replaced by the activations."""
    local = state.local.get(self.index)
    act_space = space.copy()
    if not self.additive_trq:
      del act_space[self.actkey_trq]
    act_space[self.actkey_flx] = local.flx_space
    act_space[self.actkey_ext] = local.ext_space
    return act_space
