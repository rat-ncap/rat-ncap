import abc
import dataclasses
import pathlib
import pickle
import time
import types
import typing as T

import equinox as eqx
import jax
import jax.interpreters.batching
import jax.numpy as jnp
import mujoco.mjx as mjx
import mujoco.mjx.warp.types as mjx_warp_types
import numpy as np
import tqdm

from .. import core, mjx_env


# ==================================================================================================
# Policy

PolicyState = T.TypeVar('PolicyState', contravariant=True)
"""Type of a policy's state (for the generic aliases and `TypedDict`s below; `Policy` declares its
own)."""


class Policy[PolicyState](abc.ABC):
  """Maps observations to actions for `rollout_from_policy`, possibly with state.

  The state (`get_state`/`set_state`) is everything that changes between steps, such as recurrent
  activity. Per step `i`, the rollout calls `step` (which may advance the state to `i + 1`), saves
  `get_state()` as the next state, and restores the state of step `i` with `set_state` while it
  records, so that recorders see the state that produced the action.
  """

  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    rng: jax.Array,
  ) -> None:
    """Prepares the policy for the env's spaces (vectorized), once before the rollout."""
    pass

  def reset(self, rng: jax.Array):
    """Starts new episodes, with one random key per world, `[vectorize]`."""
    pass

  @abc.abstractmethod
  def step(self, observation: core.Observation, rng: jax.Array) -> core.Action:
    """Returns the action of every world for `observation`, with random keys `[vectorize]`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_state(self) -> PolicyState:
    """Returns the current state."""
    raise NotImplementedError()

  @abc.abstractmethod
  def set_state(self, state: PolicyState) -> None:
    """Restores a state returned by `get_state`."""
    raise NotImplementedError()


class SamplingPolicy(Policy[None]):
  """A stateless policy that samples actions from the action space at each step."""
  sample_action: T.Callable[[jax.Array], core.Action]

  def __init__(
    self,
    mode: core.SampleMode = 'zero',
    jit: bool = True,
  ):
    """Initializes the policy.

    Args:
      mode: How actions are sampled (see `core.SampleMode`).
      jit: Whether to jit the sampling.
    """
    self.mode: core.SampleMode = mode
    self.jit: bool = jit

  @T.override
  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    rng: jax.Array,
  ) -> None:
    """Prepares the sampler for `action_space`."""
    def sample_action(rng: jax.Array) -> core.Action:
      """Samples the actions of every world (from the first world's key)."""
      return action_space.sample(rng[0], mode=self.mode)
    # `mjx_env._filter_jit` rather than `eqx.filter_jit`: `sample_action` is a fresh closure per
    # rollout that captures `action_space`, which `eqx.filter_jit` would retain forever.
    self.sample_action = mjx_env._filter_jit(sample_action) if self.jit else sample_action

  @T.override
  def reset(self, rng: jax.Array):
    """Does nothing: the policy is stateless."""
    pass

  @T.override
  def step(self, observation: core.Observation, rng: jax.Array) -> core.Action:
    """Returns sampled actions."""
    return self.sample_action(rng)

  @T.override
  def get_state(self) -> None:
    """Returns `None`: the policy is stateless."""
    return None

  @T.override
  def set_state(self, state: None) -> None:
    """Does nothing: the policy is stateless."""
    pass


class CyclingPolicy(Policy[int]):
  """A policy that cycles through actions: zero, min, zero, max, each for a quarter period. Its
  state is the step count."""

  def __init__(self, period: int = 40):
    """Initializes the policy with a cycle of `period` steps (rounded down to a multiple of 4)."""
    self.period = max((period // 4) * 4, 4)
    self.steps = 0

  @T.override
  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    rng: jax.Array,
  ) -> None:
    """Samples the zero, min and max actions."""
    self.action_zero = action_space.sample(rng, mode='zero')
    self.action_min = action_space.sample(rng, mode='min')
    self.action_max = action_space.sample(rng, mode='max')

  @T.override
  def reset(self, rng: jax.Array):
    """Restarts the cycle."""
    self.steps = 0

  @T.override
  def step(self, observation: core.Observation, rng: jax.Array) -> core.Action:
    """Returns the action of the cycle's current quarter."""
    phase = (self.steps % self.period) // (self.period // 4) % 4
    match phase:
      case 0 | 2:
        action = self.action_zero
      case 1:
        action = self.action_min
      case 3:
        action = self.action_max
    self.steps += 1
    return action

  @T.override
  def get_state(self) -> int:
    """Returns the step count."""
    return self.steps

  @T.override
  def set_state(self, state: int) -> None:
    """Restores the step count."""
    self.steps = state


# ==================================================================================================
# Rollout

@dataclasses.dataclass(frozen=True)
class Rollout:
  """The result of a rollout of `T` steps.

  Attributes:
    render: Rendered frames (see `mjx_env.render`), at `render_rate`.
    render_rate: Frames per second of `render`.
    record: Each recorder's values, at `record_rate`, from step 0 (the reset) on.
    record_rate: Records per second of `record`.
    reward: The reward of every step, `T + 1` entries: step 0 (the reset, reward 0) to step `T`.
  """
  render: list[np.ndarray]
  render_rate: float
  record: dict[str, list[T.Any]]
  record_rate: float
  reward: list[jax.Array]


@dataclasses.dataclass(frozen=True)
class SavedData[PolicyState]:
  """One saved step of a rollout, for `rollout_from_saved`."""
  i: int
  policy_state: PolicyState  # w[t], h[t]
  step: core.Step[mjx_env.MjxEnvState]  # s[t]
  action: core.Action  # a[t], including a[T] that is unused


@dataclasses.dataclass(frozen=True)
class RecordContext[PolicyState]:
  """What a recorder (`RecordFn`) gets at a recorded step `i`.

  Attributes:
    i: The step: 0 is the reset.
    env: The environment.
    observation_space: The observation space of one world.
    action_space: The action space of one world.
    policy: The policy, with the state of step `i` set.
    step: The env step `i` (state, observation, reward, ...).
    policy_state: The policy state of step `i`, which produced `action`.
    action: The action taken at step `i` (at the last step, computed but not taken).
    extras: A dict shared by all records of the rollout, e.g. for caching.
  """
  i: int
  env: mjx_env.MjxEnv
  observation_space: core.ObservationSpace
  action_space: core.ActionSpace
  policy: Policy[PolicyState]  # w[t], h[t]
  step: core.Step[mjx_env.MjxEnvState]  # s[t]
  policy_state: PolicyState  # w[t], h[t]
  action: core.Action  # a[t], including a[T] that is unused
  extras: dict[str, T.Any]


class EnvResetFn(T.Protocol):
  """Resets an env, as `MjxEnv.reset`."""

  def __call__(
    self,
    state: mjx_env.MjxEnvState,
    rng: jax.Array,
    *,
    subtask: mjx_env.Subtask | None = None,
  ) -> core.Step[mjx_env.MjxEnvState]:
    """Returns the reset step."""
    ...


EnvStepFn = T.Callable[[mjx_env.MjxEnvState, core.Action], core.Step[mjx_env.MjxEnvState]]
RecordFn = T.Callable[[RecordContext[PolicyState]], T.Any]
OnResetFn = T.Callable[[core.Step[mjx_env.MjxEnvState]], core.Step[mjx_env.MjxEnvState]]
OnStepFn = T.Callable[[core.Step[mjx_env.MjxEnvState]], core.Step[mjx_env.MjxEnvState]]


class RolloutFromPolicyKwargs(T.Generic[PolicyState], T.TypedDict, total=False):
  """Keyword arguments of `rollout_from_policy`."""
  # Policy
  policy: Policy[PolicyState] | core.SampleMode | T.Literal['cycle']
  # Rollout
  duration: float
  render: mjx_env.MjxRenderKwargs | None
  render_rate: float | None
  record: dict[str, RecordFn[PolicyState]]
  record_rate: float | None
  on_reset: OnResetFn | None
  on_step: OnStepFn | None
  subtask: T.Mapping[str, jax.Array] | None
  # Environment
  vectorize: int
  jit: bool
  seed: int
  # Logging
  verbose: bool


def rollout_from_policy(
  env: mjx_env.MjxEnv | tuple[mjx_env.MjxEnv, EnvResetFn, EnvStepFn, mjx_env.MjxEnvState],
  *,
  # Policy
  policy: Policy[PolicyState] | core.SampleMode | T.Literal['cycle'] = 'zero',
  # Rollout
  duration: float = 1.,
  render: mjx_env.MjxRenderKwargs | None = None,
  render_rate: float | None = 30,
  record: T.Mapping[str, RecordFn[PolicyState]] = types.MappingProxyType({}),
  record_rate: float | None = None,
  on_reset: OnResetFn | None = None,
  on_step: OnStepFn | None = None,
  subtask: T.Mapping[str, jax.Array] | None = None,
  # Environment
  vectorize: int = 1,
  jit: bool = True,
  seed: int = 0,
  # Logging
  verbose: bool = False,
) -> Rollout:
  """Runs a policy in an environment from a reset for `duration` seconds, recording as it goes.

  The rollout takes `T = ceil(duration / env_timestep)` steps. It records step 0 (the reset) to
  step `T`, so `T + 1` rewards. There is no autoreset: after an episode ends, the env keeps
  stepping with zero reward (see `core.Step`).

  Args:
    env: The env, or a tuple `(env, reset, step, state)` of an env with its (e.g. jitted) reset and
      step functions and an initialized state.
    policy: The policy, or how to sample actions (`core.SampleMode`), or 'cycle' (`CyclingPolicy`).
    duration: Duration of the rollout (s).
    render: `mjx_env.render` arguments, or `None` not to render.
    render_rate: Rendered frames per second, at most the env rate; `None` for the env rate.
    record: Recorders by name, called with a `RecordContext` at each recorded step.
    record_rate: Records per second, at most the env rate; `None` for the env rate.
    on_reset: Function applied to the reset step.
    on_step: Function applied to each step.
    subtask: The subtask of each world (see `MjxEnv.reset`).
    vectorize: Number of worlds (when `env` is an env, not a tuple).
    jit: Whether to jit the env's reset and step and the sampling policy.
    seed: Random seed of the env and policy.
    verbose: Whether to print progress and timings.

  Returns:
    The rollout's frames, records and rewards.
  """
  # Initialize environment.
  rng = jax.random.split(jax.random.key(seed), vectorize)  # [vectorize]
  if isinstance(env, mjx_env.MjxEnv):
    state = env.initialize(rng)
    env_reset = eqx.filter_jit(env.reset) if jit else env.reset
    env_step = eqx.filter_jit(env.step) if jit else env.step
  elif isinstance(env, (tuple, list)):
    env, env_reset, env_step, state = env
  else:
    raise ValueError(f'Invalid env: {env}')
  observation_space = env.observation_space(state)
  action_space = env.action_space(state)
  steps = int(np.ceil(duration / state.env_timestep))  # T

  # Initialize policy.
  if isinstance(policy, Policy):
    pass
  elif policy in ('random', 'zero', 'min', 'max', 'mid'):
    policy = T.cast(Policy, SamplingPolicy(mode=T.cast(core.SampleMode, policy), jit=jit))
  elif policy in ('cycle',):
    policy = T.cast(Policy, CyclingPolicy(period=steps))
  else:
    raise ValueError(f'Invalid policy: {policy}')
  policy.initialize(observation_space, action_space, rng[0])
  policy.reset(rng)
  policy_step = policy.step

  # Run rollout.
  env_rate = 1 / state.env_timestep
  render_rate = min(render_rate or env_rate, env_rate)
  record_rate = min(record_rate or env_rate, env_rate)
  render_data: list[mjx_env.MjxEnvState] = []
  record_data = {k: [] for k in record.keys()}
  reward_data: list[jax.Array] = []
  renders = 0
  records = 0
  observation_space_one = observation_space.unvectorize()
  action_space_one = action_space.unvectorize()
  if verbose:
    print(
      f'Rollout: {duration:.1f} sec, {steps} steps, render @ {render_rate:.1f} Hz, '
      f'record @ {record_rate:.1f} Hz'
    )

  rollout_time = time.perf_counter()
  reset_time = time.perf_counter()
  step = env_reset(state, rng, subtask=subtask)
  if on_reset: step = on_reset(step)  # s[0]
  policy_state = policy.get_state()  # w[0], h[0]
  if verbose: print(f'  reset: {time.perf_counter() - reset_time:.4f} sec')

  extras = {}
  rng_step = jax.vmap(jax.random.split)
  for i in tqdm.tqdm(range(steps + 1), disable=not verbose):  # [0, ..., T]
    step_time = time.perf_counter()
    t = i * state.env_timestep
    rng, rng1 = rng_step(rng).T
    action = policy_step(step.observation, rng1)  # a[i]
    next_policy_state = policy.get_state()  # w[i+1], h[i+1]
    policy.set_state(policy_state)  # w[i], h[i]
    if render and renders <= t * render_rate:
      render_data.append(step.state)
      renders += 1
    if record and records <= t * record_rate:
      ctx = RecordContext(
        i=i, # elapsed steps
        env=env,
        observation_space=observation_space_one,
        action_space=action_space_one,
        policy=policy,  # w[i], h[i]
        step=step,  # s[i]
        policy_state=policy_state,  # w[i], h[i]
        action=action,  # a[i], including a[T] that is unused
        extras=extras,
      )
      for name, record_fn in record.items():
        record_data[name].append(record_fn(ctx))
      records += 1
    reward_data.append(step.reward)  # r[i], including r[0] = 0 since no action taken yet
    if i == steps: break
    step = env_step(step.state, action)  # s[i+1]
    if on_step: step = on_step(step)
    policy_state = next_policy_state  # w[i+1], h[i+1]
    policy.set_state(policy_state)
    if verbose: print(f'  step {i}/{steps}: {time.perf_counter() - step_time:.4f} sec')
  if verbose: print(f'  total: {time.perf_counter() - rollout_time:.4f} sec')

  # Run rendering.
  render_frames = []
  if render:
    if verbose: print(f'  render ({len(render_data)} frames):', end=' ')
    render_time = time.perf_counter()
    render_frames = mjx_env.render(render_data, **render)
    if verbose: print(f'{time.perf_counter() - render_time:.4f} sec')

  return Rollout(
    render=render_frames,
    render_rate=render_rate,
    record=record_data,
    record_rate=record_rate,
    reward=reward_data,
  )


class RolloutFromSavedKwargs(T.Generic[PolicyState], T.TypedDict, total=False):
  """Keyword arguments of `rollout_from_saved`."""
  duration: float
  render: mjx_env.MjxRenderKwargs | None
  render_rate: float
  record: dict[str, RecordFn[PolicyState]]
  record_rate: float | None
  verbose: bool


def rollout_from_saved(
  env: mjx_env.MjxEnv,
  policy: Policy[PolicyState],
  data: list[SavedData[PolicyState]],
  observation_space: core.ObservationSpace,
  action_space: core.ActionSpace,
  *,
  # Rollout
  duration: float = 1.,
  render: mjx_env.MjxRenderKwargs | None = None,
  render_rate: float | None = 30,
  record: T.Mapping[str, RecordFn[PolicyState]] = types.MappingProxyType({}),
  record_rate: float | None = None,
  # Logging
  verbose: bool = False,
) -> Rollout:
  """Replays saved steps of a rollout (`SavedData`) for recording and rendering, without stepping.

  Arguments are as in `rollout_from_policy`. Rendering stops once the rendered worlds' episodes have
  ended.
  """
  # Run rollout.
  state = data[0].step.state
  steps = min(int(np.ceil(duration / state.env_timestep)), len(data) - 1)
  env_rate = 1 / state.env_timestep
  render_rate = min(render_rate or env_rate, env_rate)
  record_rate = min(record_rate or env_rate, env_rate)
  render_data: list[mjx_env.MjxEnvState] = []
  record_data = {k: [] for k in record.keys()}
  reward_data: list[jax.Array] = []
  renders = 0
  records = 0

  extras = {}
  terminated = False
  for i in range(steps + 1):
    t = i * state.env_timestep
    step = data[i].step
    if render and renders <= t * render_rate and not terminated:
      render_data.append(step.state)
      renders += 1
      index = render.get('index', 0)
      terminated = bool(step.terminated[jnp.asarray(index)].all())
    if record and records <= t * record_rate:
      policy_state = data[i].policy_state
      policy.set_state(policy_state)
      action = data[i].action
      ctx = RecordContext(
        i=i,  # elapsed steps
        env=env,
        observation_space=observation_space,
        action_space=action_space,
        policy=policy,  # w[i], h[i]
        step=step,  # s[i]
        action=action,  # a[i], including a[T] that is unused
        policy_state=policy_state,  # w[i], h[i]
        extras=extras,
      )
      for name, record_fn in record.items():
        record_data[name].append(record_fn(ctx))
      records += 1
    reward_data.append(step.reward)

  # Run rendering.
  render_frames = []
  if render:
    if verbose: print(f'  render ({len(render_data)} frames):', end=' ')
    render_time = time.perf_counter()
    render_frames = mjx_env.render(render_data, **render)
    if verbose: print(f'{time.perf_counter() - render_time:.4f} sec')

  return Rollout(
    render=render_frames,
    render_rate=render_rate,
    record=record_data,
    record_rate=record_rate,
    reward=reward_data,
  )


# ==================================================================================================
# Save / Load

class SerializedSteps(T.TypedDict):
  """The contents of a file saved by `save_steps`."""
  spec: str  # mj_spec XML
  vectorize: int
  static_axes: dict[str, dict[str, bool]]
  static: dict[str, dict[str, T.Any]]
  dynamic: list[core.Step[mjx_env.MjxEnvState]]


class _MockJAXImpl(eqx.Module):
  """A pickleable subset of `mjx.DataJAX` (too large to save whole): the fields needed for contact
  forces and tendon velocities."""
  # Fields used to calculate contact forces.
  contact: mjx.Contact
  efc_force: jax.Array
  ten_velocity: jax.Array


class _MockWarpImpl(eqx.Module):
  """A pickleable subset of `DataWarp` (too large to save whole): the fields needed for contact
  forces and tendon velocities."""
  # Fields used to calculate contact forces (see `mjx_env.contact_force_geom_world`).
  contact__dim: jax.Array
  contact__efc_address: jax.Array
  contact__frame: jax.Array
  contact__friction: jax.Array
  contact__geom: jax.Array
  contact__worldid: jax.Array
  nacon: jax.Array
  efc__force: jax.Array
  ten_velocity: jax.Array

  @property
  def shape(self) -> tuple[int, ...]:
    """Batch shape used by `jax.vmap` to determine the axis size, as for `DataWarp`."""
    return self.efc__force.shape


# Register `jax.vmap` handlers mirroring those of `DataWarp`, which keep the contact buffer shared
# by all worlds unbatched.
def _mock_warp_impl_to_elt(cont, _, d: _MockWarpImpl, axis) -> _MockWarpImpl:
  """Unbatches the batched fields of `d` (the `jax.vmap` "to element" handler)."""
  return _MockWarpImpl(**{
    f.name: (
      getattr(d, f.name) if f.name in mjx_warp_types.DATA_NON_VMAP
      else cont(getattr(d, f.name), axis)
    )
    for f in dataclasses.fields(d)
  })


def _mock_warp_impl_from_elt(cont, axis_size, d: _MockWarpImpl, axis_dest) -> _MockWarpImpl:
  """Batches the batched fields of `d` (the `jax.vmap` "from element" handler)."""
  return _MockWarpImpl(**{
    f.name: (
      getattr(d, f.name) if f.name in mjx_warp_types.DATA_NON_VMAP
      else cont(axis_size, getattr(d, f.name), axis_dest)
    )
    for f in dataclasses.fields(d)
  })


jax.interpreters.batching.register_vmappable(
  _MockWarpImpl, int, int, _mock_warp_impl_to_elt, _mock_warp_impl_from_elt, None
)


_DEFAULT_STATIC_AXES = types.MappingProxyType({'state': {'mjx_model': True}})


def serialize_steps(
  steps: T.Sequence[core.Step[mjx_env.MjxEnvState]],
  static_axes: T.Mapping[str, T.Any] | None = _DEFAULT_STATIC_AXES,
) -> SerializedSteps:
  """Converts steps to a pickleable dict, separating the data shared by all steps from the per-step
  data, and leaving out the recomputable parts of `mjx_data` to keep it small.

  Args:
    steps: The steps.
    static_axes: Axes to save once rather than per step (stored, not yet used).

  Returns:
    The serialized steps.
  """
  # Save MjxEnvState.mj_spec separately, since it does not change and cannot be pickled.
  spec = steps[0].state.mj_spec.to_xml()
  vectorize = steps[0].state.vectorize
  steps = [
    dataclasses.replace(step, state=step.state.replace(
      mj_spec=None,
      mj_model=None,
    )) for step in steps
  ]

  static = {
    'state': {
      'mjx_model': steps[0].state.mjx_model,
    }
  }

  def filter_impl(_impl: mjx.DataCPP | mjx.DataJAX | mjx_warp_types.DataWarp):
    """Returns the saved subset of `mjx_data._impl` (`None` for the C implementation)."""
    if isinstance(_impl, mjx.DataJAX):
      return _MockJAXImpl(
        contact=_impl.contact,
        efc_force=_impl.efc_force,
        ten_velocity=_impl.ten_velocity
      )
    if isinstance(_impl, mjx_warp_types.DataWarp):
      return _MockWarpImpl(
        contact__dim=_impl.contact__dim,
        contact__efc_address=_impl.contact__efc_address,
        contact__frame=_impl.contact__frame,
        contact__friction=_impl.contact__friction,
        contact__geom=_impl.contact__geom,
        contact__worldid=_impl.contact__worldid,
        nacon=_impl.nacon,
        efc__force=_impl.efc__force,
        ten_velocity=_impl.ten_velocity,
      )
    return None
  dynamic = [
    dataclasses.replace(step, state=step.state.replace(
      mjx_model=None,
      mjx_data=step.state.mjx_data.replace(
        _impl=filter_impl(step.state.mjx_data._impl),
      )
    )) for step in steps
  ]

  return {
    'spec': spec,
    'vectorize': vectorize,
    'static_axes': dict(static_axes or {}),
    'static': static,
    'dynamic': dynamic,
  }


def save_steps(
  filepath: str | pathlib.Path,
  steps: T.Sequence[core.Step[mjx_env.MjxEnvState]],
  static_axes: T.Mapping[str, T.Any] | None = _DEFAULT_STATIC_AXES,
) -> None:
  """Saves steps to a file (see `serialize_steps`), for `load_steps`.

  Args:
    filepath: Path of the file to write.
    steps: The steps.
    static_axes: Axes to save once rather than per step (stored, not yet used).
  """
  with open(filepath, 'wb') as f:
    pickle.dump(serialize_steps(steps, static_axes), f)


def load_steps(
  filepath: str | pathlib.Path,
  env: mjx_env.MjxEnv,
) -> list[core.Step[mjx_env.MjxEnvState]]:
  """Loads steps saved by `save_steps`.

  The model is rebuilt from `env`'s current spec, not from the spec saved in the file, so `env`
  must be configured as when the steps were saved.

  Args:
    filepath: Path of the file to read.
    env: The env that produced the steps, whose spec and backend rebuild the model.

  Returns:
    The steps, with the model re-inserted. `mjx_data` holds only the saved subset of its
    implementation fields.
  """
  # Equinox (0.13.0) registers every `eqx.Module` in a global `_currently_initialising` set from
  # `Module.__new__`, and only unregisters it in `Module.__call__` after `__init__`. Unpickling
  # calls `__new__` but not `__call__`, so every loaded module (e.g. each step's `MjxEnvState` and
  # its arrays) would stay referenced forever, leaking ~0.75 GiB per load of a typical rollout.
  # Unregister the modules added during the load; none of them are mid-`__init__`.
  initialising = getattr(eqx._module._module, '_currently_initialising', None)
  initialising_ids = set(initialising._dict) if initialising is not None else set()
  try:
    with open(filepath, 'rb') as f:
      rollout_data = pickle.load(f)
  finally:
    if initialising is not None:
      for id_key in set(initialising._dict) - initialising_ids:
        del initialising._dict[id_key]

  static = rollout_data['static']
  dynamic = rollout_data['dynamic']
  vectorize = rollout_data['vectorize']

  # Rebuild the model from the env's current spec (the saved `spec` XML is not read).
  spec = env.task.mj_spec
  mj_model = spec.compile()
  mjx_model = mjx.put_model(mj_model, impl=mjx_env.PHYSICS_BACKENDS[env.backend].impl)

  # Reinsert static fields.
  steps = [
    dataclasses.replace(step, state=step.state.replace(
      mj_spec=spec,
      mj_model=mj_model,
      mjx_model=mjx_model,
    )) for step in dynamic
  ]
  return steps
