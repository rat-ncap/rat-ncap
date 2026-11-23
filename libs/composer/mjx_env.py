import abc
import collections
import functools
import os
import types
import typing as T
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import mujoco.mjx.warp.render_context as mjxrc
import mujoco.mjx.warp.types as mjx_warp_types
import numpy as np

from . import core, mujoco_physics

try:
  import warp as wp
  wp.config.quiet = True
  _warp_available = True
except ImportError:
  _warp_available = False

# ==================================================================================================
# State and entities


class MjxEnvState(core.BaseEnvState):
  """State of a vectorized `MjxEnv`: `vectorize` worlds that share one compiled model.

  Arrays have a leading `[vectorize]` axis, except the `mjx_model` fields that are not per-world
  (all but `MjxTask.mj_model_vecfields`), which are shared by all worlds.

  Attributes:
    mj_spec: Model specification the model was compiled from.
    mj_model: Compiled MuJoCo model, shared by all worlds.
    mjx_model: MJX model.
    mjx_data: MJX data of each world.
    env_timestep: Duration of one environment step (s).
    sim_timestep: Duration of one simulation substep (s).
    sim_substeps: Simulation substeps per environment step.
    vectorize: Number of worlds.
    rng: Random key of each world's episode.
    steps: Environment steps taken in each world's episode.
    substeps: Simulation substeps taken in each world's episode.
    terminated: Whether each world's episode has ended (sticky, see `core.Step`).
    local: Local states of the entities and transforms.
  """
  mj_spec: mj.MjSpec = eqx.field(static=True)
  mj_model: mj.MjModel = eqx.field(static=True)
  mjx_model: mjx.Model
  mjx_data: mjx.Data
  env_timestep: float = eqx.field(static=True)
  sim_timestep: float = eqx.field(static=True)
  sim_substeps: int = eqx.field(static=True)
  vectorize: int = eqx.field(static=True)
  rng: jax.Array  # [vectorize], key
  steps: jax.Array  # [vectorize], int
  substeps: jax.Array  # [vectorize], int
  terminated: jax.Array  # [vectorize], bool
  local: core.State


class MjxEntity[LocalState: core.BaseLocalState | None](core.Entity[MjxEnvState, LocalState]):
  """An entity with a MuJoCo model part, such as a body or arena.

  Attributes:
    mj_spec: The entity's model specification, attached into its parent's by the task.
  """
  children: collections.OrderedDict[str, 'MjxEntity[T.Any]']
  transforms: list[core.Transform]
  index: eqx.nn.StateIndex[LocalState]
  mj_spec: mj.MjSpec = eqx.field(static=True)

  def __init__(
    self,
    children: T.Mapping[str, 'MjxEntity[T.Any] | None'] = types.MappingProxyType({}),
    transforms: T.Sequence[core.Transform | T.Literal[False] | None] = (),
    local_state: LocalState = None,  # type: ignore  # See `core.Transform.__init__`.
    mj_spec: mj.MjSpec | None = None,
  ):
    """Initializes the entity.

    Args:
      children: Child entities by name. `None` entries are dropped.
      transforms: Transforms, innermost first. `None` and `False` entries are dropped.
      local_state: Initial local state, or `None` if the entity has none.
      mj_spec: The entity's model specification. Defaults to an empty one.
    """
    super().__init__(
      children=children,
      transforms=transforms,
      local_state=local_state,
    )
    self.mj_spec = mj_spec if mj_spec is not None else mj.MjSpec()


# ==================================================================================================
# Task

# Subtasks are currently a MjxTask concept, so provision and enforcement is not core-level like
# actions and observations are.
Subtask = T.Mapping[str, jax.Array]
"""Values that select an episode's variant of the task, such as a target speed."""
SubtaskSpace = core.Dict
MjxTaskRewards = dict[str, tuple[jax.Array, jax.Array]]
"""Reward terms by name: `(weight, value)`."""
StepReturns = tuple[core.Observation, jax.Array, jax.Array, jax.Array, core.Metrics]
"""`(observation, reward, terminated, truncated, metrics)` of one world's step."""


class MjxTask[LocalState: core.BaseLocalState | None](core.Entity[MjxEnvState, LocalState]):
  """The root entity of an `MjxEnv`: owns the model, the timesteps, and the episode logic.

  Subclasses override the hooks below. Within `MjxEnv.reset`, `on_reset` runs once. Within
  `MjxEnv.step`, they run in this order:
  1. `on_step_start`;
  2. per simulation substep: `on_substep_start`, the action is applied, the physics steps, then
     `on_substep_end`. After each substep but the last, an observation is taken and discarded, so
     stateful observations update every substep;
  3. `on_step_end`;
  4. the observation, then `get_terminated`, `get_rewards`, `get_reward` and `get_metrics`.

  Attributes:
    mj_spec: The model specification of the whole environment.
    mj_model_vecfields: `mjx.Model` fields that differ per world (e.g. randomized masses). Other
      fields are shared by all worlds.
    mj_model_nconmax: Maximum contacts per world, or `None` for MJX's default.
    mj_model_njmax: Maximum constraint rows per world, or `None` for MJX's default.
    env_timestep: Duration of one environment step (s).
    sim_timestep: Duration of one simulation substep (s).
    sim_substeps: Simulation substeps per environment step.
  """
  children: collections.OrderedDict[str, 'MjxEntity[T.Any]']
  transforms: list[core.Transform]
  index: eqx.nn.StateIndex[LocalState]
  mj_spec: mj.MjSpec = eqx.field(static=True)
  mj_model_vecfields: list[str] = eqx.field(static=True)
  mj_model_nconmax: int | None = eqx.field(static=True)
  mj_model_njmax: int | None = eqx.field(static=True)
  env_timestep: float = eqx.field(static=True)
  sim_timestep: float = eqx.field(static=True)
  sim_substeps: int = eqx.field(static=True)

  def __init__(
    self,
    children: T.Mapping[str, 'MjxEntity[T.Any] | None'] = types.MappingProxyType({}),
    transforms: T.Sequence[core.Transform | T.Literal[False] | None] = (),
    local_state: LocalState = None,  # type: ignore  # See `core.Transform.__init__`.
    mj_spec: mj.MjSpec | None = None,
    mj_model_vecfields: T.Sequence[str] = (),
    mj_model_nconmax: int | None = None,
    mj_model_njmax: int | None = None,
    env_timestep: float = 0.01,
    sim_timestep: float = 0.001,
  ):
    """Initializes the task.

    Args:
      children: Child entities by name. `None` entries are dropped.
      transforms: Transforms, innermost first. `None` and `False` entries are dropped.
      local_state: Initial local state, or `None` if the task has none.
      mj_spec: The model specification. Defaults to an empty one.
      mj_model_vecfields: `mjx.Model` fields that differ per world.
      mj_model_nconmax: Maximum contacts per world, or `None` for MJX's default.
      mj_model_njmax: Maximum constraint rows per world, or `None` for MJX's default.
      env_timestep: Duration of one environment step (s). Must be a whole multiple of
        `sim_timestep`.
      sim_timestep: Duration of one simulation substep (s), MuJoCo's `opt.timestep`.
    """
    super().__init__(
      children=children,
      transforms=transforms,
      local_state=local_state,
    )
    self.mj_spec = mj_spec if mj_spec is not None else mj.MjSpec()
    self.mj_model_vecfields = list(mj_model_vecfields)
    self.mj_model_nconmax = mj_model_nconmax
    self.mj_model_njmax = mj_model_njmax

    # Validate that the environment timestep is an integer multiple of the simulation timestep.
    assert env_timestep >= sim_timestep > 0
    self.env_timestep = env_timestep
    self.sim_timestep = sim_timestep
    self.sim_substeps = int(round(env_timestep / sim_timestep))
    assert abs(env_timestep / sim_timestep - self.sim_substeps) < 1e-3

    # Set the simulation timestep and maximum rendering resolution at 4K.
    self.mj_spec.option.timestep = self.sim_timestep
    self.mj_spec.visual.global_.offwidth = 3840
    self.mj_spec.visual.global_.offheight = 2160

  def subtask_space(self, state: MjxEnvState) -> SubtaskSpace:
    """Returns the space of the `subtask` passed to `on_reset` (one world). Defaults to empty."""
    return core.Dict()

  def on_reset(self, state: MjxEnvState, subtask: Subtask) -> MjxEnvState:
    """Starts an episode (one world), after `mjx_data` and the local states are reset.

    Commonly sets the initial positions and velocities, randomizes the MJX model or data, or
    recompiles the model specification. `mjx_data`'s derived fields are recomputed afterwards.
    """
    return state

  def on_step_start(self, state: MjxEnvState, action: core.Action) -> MjxEnvState:
    """Runs at the start of each environment step (one world), after `steps` is incremented."""
    return state

  def on_substep_start(self, state: MjxEnvState, action: core.Action) -> MjxEnvState:
    """Runs at the start of each simulation substep (one world), before the action is applied."""
    return state

  def on_substep_end(self, state: MjxEnvState, action: core.Action) -> MjxEnvState:
    """Runs at the end of each simulation substep (one world), after the physics step."""
    return state

  def on_step_end(self, state: MjxEnvState, action: core.Action) -> MjxEnvState:
    """Runs at the end of each environment step (one world), before the observation."""
    return state

  def get_terminated(self, state: MjxEnvState) -> jax.Array:
    """Returns whether the episode ends at this step (one world, bool). Defaults to never."""
    return jnp.zeros((), dtype=bool)

  def get_rewards(self, state: MjxEnvState, terminated: jax.Array) -> MjxTaskRewards:
    """Returns the reward terms of this step (one world). Defaults to none."""
    return {}

  def get_reward(self, rewards: MjxTaskRewards) -> jax.Array:
    """Returns the step's reward from its terms. Defaults to the weighted sum."""
    return sum((w * v for w, v in rewards.values()), jnp.zeros(()))

  def get_metrics(
    self,
    state: MjxEnvState,
    terminated: jax.Array,
    rewards: MjxTaskRewards,
  ) -> core.Metrics:
    """Returns the step's metrics (one world). Defaults to each reward term's weight and value."""
    return (
      {f'reward/weight/{k}': jnp.asarray(w) for k, (w, v) in rewards.items()} |
      {f'reward/value/{k}': v for k, (w, v) in rewards.items()}
    )


# ==================================================================================================
# Vectorization

# Default `mjx.Model` fields of `_mjx_model_axes` that may differ per world.
_MJX_MODEL_VECTORIZE_FIELDS = [
  # Visual randomization (required to copy into Madrona buffers).
  'geom_rgba',
  'geom_matid',
  'geom_size',
  'light_pos',
  'light_dir',
  'light_type',
  'light_castshadow',
  'light_cutoff',
  # Geom randomization.
  'geom_friction',
  # Dof randomization.
  'dof_frictionloss',
  'dof_armature',
  'dof_damping',
  # Body randomization.
  'body_pos',
  'body_quat',
  'body_mass',
  'body_ipos',
  'body_iquat',
  'body_inertia',
  # Camera randomization.
  'cam_pos',
  'cam_quat',
]


def _mjx_model_axes(
  mjx_model: mjx.Model,
  fields: T.Sequence[str] = _MJX_MODEL_VECTORIZE_FIELDS,
) -> mjx.Model:
  """Returns `jax.vmap` axes for `mjx_model`: axis 0 for the per-world `fields`, `None` (shared)
  for the rest."""
  axes = jax.tree.map(lambda x: None, mjx_model)
  axes = axes.tree_replace({field: 0 for field in fields})
  return axes


def _filter_jit[**P, R](fn: T.Callable[P, R]) -> T.Callable[P, R]:
  """Replacement for `eqx.filter_jit` whose compilation cache is owned by the returned wrapper.

  `eqx.filter_jit` (Equinox 0.13.0) never frees a jitted closure or anything it captures, even with
  `jax.clear_caches()`, so jitting a fresh closure per env or rollout leaks memory. This wrapper has
  the same filtering semantics (array leaves of the inputs and outputs are traced; all other leaves
  are static and part of the cache key), but keeps one `jax.jit` per input signature in its own
  closure, so everything is freed with the wrapper. It does not support `eqx.filter_jit` extras such
  as donation or `.lower()`.
  """
  # Per input signature: the jitted function, and the output layout recorded when it was traced as
  # `[treedef, is_dynamic, static_leaves]`.
  compiled: dict[T.Any, tuple[T.Callable[[list[T.Any]], list[T.Any]], list[T.Any]]] = {}

  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    """Calls the jitted `fn` for this input signature, tracing it on first use."""
    # Flatten the inputs only once and split leaves by position, and rebuild the output by
    # unflattening its recorded treedef. Flattening `mjx.Model` re-hashes its NumPy metadata, so
    # `eqx.partition`/`eqx.combine` on the state per call would cost ~10x the jitted function.
    leaves, treedef = jax.tree.flatten((args, kwargs))
    is_dynamic = tuple(eqx.is_array(x) for x in leaves)
    dynamic = [x for x, d in zip(leaves, is_dynamic) if d]
    # Key on input shapes/dtypes too, since the static part of the output may depend on them.
    key = (
      treedef,
      is_dynamic,
      tuple((x.shape, x.dtype) for x in dynamic),
      tuple(x for x, d in zip(leaves, is_dynamic) if not d),
    )
    if key not in compiled:
      static = [None if d else x for x, d in zip(leaves, is_dynamic)]
      out_layout: list[T.Any] = []

      def traced(dynamic: list[T.Any]) -> list[T.Any]:
        """`fn` on the dynamic input leaves, returning its dynamic output leaves."""
        dynamic_iter = iter(dynamic)
        args, kwargs = jax.tree.unflatten(
          treedef, [next(dynamic_iter) if d else x for x, d in zip(static, is_dynamic)]
        )
        out_leaves, out_treedef = jax.tree.flatten(fn(*args, **kwargs))
        out_is_dynamic = [eqx.is_array(x) for x in out_leaves]
        out_static = [None if d else x for x, d in zip(out_leaves, out_is_dynamic)]
        out_layout[:] = [out_treedef, out_is_dynamic, out_static]
        return [x for x, d in zip(out_leaves, out_is_dynamic) if d]

      compiled[key] = (jax.jit(traced), out_layout)
    jitted, out_layout = compiled[key]
    out_dynamic = iter(jitted(dynamic))
    out_treedef, out_is_dynamic, out_static = out_layout
    return jax.tree.unflatten(
      out_treedef, [next(out_dynamic) if d else x for x, d in zip(out_static, out_is_dynamic)]
    )

  return wrapper


WORLD_AXIS = 'mjx_world'
"""Name of the `jax.vmap` axis over worlds in `vmap_state_update` and `vmap_state`. Per-world code
that needs its world index (e.g. to select its contacts from the 'warp' backend's contact buffer
shared by all worlds) can get it with `jax.lax.axis_index(WORLD_AXIS)`."""


def vmap_state_update[*Args, Returns](
  func: T.Callable[[MjxEnvState, *Args], tuple[MjxEnvState, Returns]],
  mj_model_vecfields: T.Sequence[str] = (),
  in_axes: T.Sequence[T.Any] = (),
  out_axes: T.Any = 0,
  jit: bool = False,
) -> T.Callable[[MjxEnvState, *Args], tuple[MjxEnvState, Returns]]:
  """Vectorizes a per-world function that takes and returns an `MjxEnvState` over the worlds.

  `mjx.Model` and `mjx.Data` contain fields that are not, or need not be, vectorized, so a naive
  `jax.vmap` can take prohibitive compilation time and memory. This maps only the per-world parts:
  the state's arrays, the `mj_model_vecfields` of `mjx_model`, and `mjx_data` (with MJX's own vmap
  rules for the 'warp' backend's shared buffers).

  Args:
    func: Per-world function `func(state, *args) -> (state, returns)`.
    mj_model_vecfields: `mjx.Model` fields that differ per world.
    in_axes: `jax.vmap` axes of `args`, one per argument.
    out_axes: `jax.vmap` axes of the array leaves of `returns`, as a pytree prefix. Non-array
      leaves are always unmapped.
    jit: Whether to jit the vectorized function (with `_filter_jit`).

  Returns:
    The vectorized function, with the same signature on batched states.
  """
  def vmapped_func(state: MjxEnvState, *args: *Args) -> tuple[MjxEnvState, Returns]:
    """`func` vectorized over the worlds of `state`."""
    def func_one(
      state_dynamic: MjxEnvState,
      state_static: MjxEnvState,
      mjx_model: mjx.Model,
      mjx_data: mjx.Data,
      *args: *Args,
    ) -> tuple[T.Any, T.Any, mjx.Model, mjx.Data, T.Any, T.Any]:
      """`func` on one world, with the state recombined inside and partitioned again outside."""
      # Recombine dynamic/static state and reinsert MJX fields.
      state = eqx.combine(state_dynamic, state_static)
      state = state.replace(mjx_model=mjx_model, mjx_data=mjx_data)
      # Call function using singleton logic.
      state, returns = func(state, *args)
      # Separate dynamic/static state and extract MJX fields.
      mjx_model, mjx_data = state.mjx_model, state.mjx_data
      state = state.replace(mjx_model=None, mjx_data=None)
      state_dynamic, state_static = eqx.partition(state, eqx.is_array)
      # Separate dynamic/static returns: non-array leaves (e.g. the static fields of returned
      # spaces) cannot cross the vmap boundary as mapped outputs.
      returns_dynamic, returns_static = eqx.partition(returns, eqx.is_array)
      return state_dynamic, state_static, mjx_model, mjx_data, returns_dynamic, returns_static
    
    # Separate dynamic/static state and extract MJX fields outside of vmap.
    mjx_model, mjx_data = state.mjx_model, state.mjx_data
    mjx_model_axes = _mjx_model_axes(mjx_model, fields=mj_model_vecfields)
    state = state.replace(mjx_model=None, mjx_data=None)
    state_dynamic, state_static = eqx.partition(state, eqx.is_array)
    # Vmap over partitioned state.
    state_dynamic, state_static, mjx_model, mjx_data, returns_dynamic, returns_static = jax.vmap(
      func_one,
      in_axes=(0, None, mjx_model_axes, 0, *in_axes),
      out_axes=(0, None, mjx_model_axes, 0, out_axes, None),
      axis_name=WORLD_AXIS,
    )(state_dynamic, state_static, mjx_model, mjx_data, *args)
    # Recombine dynamic/static state and reinsert MJX fields after vmap.
    state = eqx.combine(state_dynamic, state_static)
    state = state.replace(mjx_model=mjx_model, mjx_data=mjx_data)
    returns = eqx.combine(returns_dynamic, returns_static)
    return state, returns
  
  if jit:
    # `_filter_jit` rather than `eqx.filter_jit`: `vmapped_func` is a fresh closure per call (e.g.
    # per rollout in record functions) that captures `func` and thus the env, which
    # `eqx.filter_jit` would retain forever (~0.2 GiB per rollout video).
    return _filter_jit(vmapped_func)
  return vmapped_func


def vmap_state[*Args, Returns](
  func: T.Callable[[MjxEnvState, *Args], Returns],
  mj_model_vecfields: T.Sequence[str] = (),
  in_axes: T.Sequence[T.Any] = (),
  out_axes: T.Any = 0,
  jit: bool = False,
) -> T.Callable[[MjxEnvState, *Args], Returns]:
  """Vectorizes a per-world function that takes an `MjxEnvState` (and leaves it unchanged) over
  the worlds. Arguments as in `vmap_state_update`."""
  def func_with_state(state: MjxEnvState, *args: *Args) -> tuple[MjxEnvState, Returns]:
    """`func`, returning the state unchanged."""
    return state, func(state, *args)

  vmapped = vmap_state_update(
    func_with_state,
    mj_model_vecfields=mj_model_vecfields,
    in_axes=in_axes,
    out_axes=out_axes,
    jit=jit,
  )

  def vmapped_returns(state: MjxEnvState, *args: *Args) -> Returns:
    """The vectorized `func`'s returns."""
    return vmapped(state, *args)[1]

  return vmapped_returns


# ==================================================================================================
# Physics backends

Backend = T.Literal['jax', 'warp', 'mujoco']
"""Physics backend of `MjxEnv`:
- 'jax': MJX's JAX physics (GPU; compiling it on CPU is impractical: hours for one env step).
- 'warp': MJX's Warp physics (GPU only).
- 'mujoco': native C MuJoCo through a host callback (`mujoco_physics`), with all task code still in
  JAX. Compiles in seconds and runs on CPU (rollouts without a GPU). Uses MJX's 'jax' data layout.
"""


class PhysicsBackend(abc.ABC):
  """Steps the `mjx.Data` of one environment for `MjxEnv` (inside `vmap_state_update`).

  Attributes:
    impl: The MJX implementation whose `mjx.Model` and `mjx.Data` layout the backend uses.
  """
  impl: T.Literal['jax', 'warp']

  @abc.abstractmethod
  def check_supported(self, env: 'MjxEnv') -> None:
    """Raises `AssertionError` if the backend cannot run `env`: a missing dependency, or an env
    feature the backend does not support."""
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` after one simulation substep."""
    raise NotImplementedError()

  @abc.abstractmethod
  def forward(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` with its derived fields recomputed."""
    raise NotImplementedError()


class MjxBackend(PhysicsBackend):
  """MJX's own physics: the 'jax' and 'warp' backends."""

  def __init__(self, impl: T.Literal['jax', 'warp']):
    """Initializes the backend for MJX implementation `impl`."""
    self.impl = impl

  @T.override
  def check_supported(self, env: 'MjxEnv') -> None:
    """Raises if the 'warp' backend is requested but Warp is not installed."""
    assert self.impl != 'warp' or _warp_available, (
      'Warp backend is not available. Please install Warp to use it.'
    )

  @T.override
  def step(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` after `mjx.step`."""
    return mjx.step(state.mjx_model, state.mjx_data)

  @T.override
  def forward(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` after `mjx.forward`."""
    return mjx.forward(state.mjx_model, state.mjx_data)


class MujocoBackend(PhysicsBackend):
  """Native C MuJoCo physics (`mujoco_physics.MujocoPhysics`) on MJX's 'jax' layout: the 'mujoco'
  backend."""
  impl = 'jax'

  @T.override
  def check_supported(self, env: 'MjxEnv') -> None:
    """Raises if the task has per-world model fields: every world steps the same `mj_model`."""
    assert not env.task.mj_model_vecfields, (
      "The 'mujoco' backend does not support per-world model fields (`mj_model_vecfields`)."
    )

  @T.override
  def step(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` after `mj_step`."""
    return mujoco_physics.get_physics(state.mj_model).step(state.mjx_data)

  @T.override
  def forward(self, state: 'MjxEnvState') -> mjx.Data:
    """Returns `state.mjx_data` after `mj_forward`."""
    return mujoco_physics.get_physics(state.mj_model).forward(state.mjx_data)


PHYSICS_BACKENDS: T.Mapping[Backend, PhysicsBackend] = types.MappingProxyType(T.cast(
  dict[Backend, PhysicsBackend],
  dict(
    jax=MjxBackend('jax'),
    warp=MjxBackend('warp'),
    mujoco=MujocoBackend(),
  ),
))
"""The physics of each backend."""


# ==================================================================================================
# Environment


class MjxEnv(core.Env[MjxEnvState]):
  """A vectorized MuJoCo environment of an `MjxTask`, simulated by MJX or native MuJoCo.

  Every method takes and returns a state of `vectorize` worlds (the size of the `rng` passed to
  `initialize`). The task's hooks run per world, vectorized with `vmap_state_update`.

  Attributes:
    task: The task, with its state indices initialized.
    truncate: Episode length in steps after which `truncated` is set, or `None` for no limit.
    backend: The physics backend (see `Backend`).
  """
  task: MjxTask
  truncate: int | None = eqx.field(static=True)
  backend: Backend = eqx.field(static=True)

  def __init__(
    self,
    task: MjxTask,
    *,
    truncate: int | None = None,
    backend: Backend = 'jax',
  ):
    """Initializes the environment.

    Args:
      task: The task.
      truncate: Episode length in steps after which `truncated` is set, or `None` for no limit.
      backend: The physics backend (see `Backend`).

    Raises:
      AssertionError: If `truncate` is not positive, or the backend is unknown or cannot run the
        task.
    """
    assert truncate is None or truncate > 0
    assert backend in PHYSICS_BACKENDS, f'Unknown backend {backend!r}.'

    # Initialize all state index in the task pytree.
    self.task = core.initialize_state_index(task)
    self.truncate = truncate
    self.backend = backend
    self.physics.check_supported(self)

  @property
  def physics(self) -> PhysicsBackend:
    """The physics of `backend`."""
    return PHYSICS_BACKENDS[self.backend]

  @T.override
  def initialize(self, rng: jax.Array) -> MjxEnvState:
    """Compiles the model and returns the initial state of `vectorize` worlds, before `reset`.

    Args:
      rng: Random key of each world, `[vectorize]`.
    """
    # Extract vectorize size.
    vectorize = rng.shape[0]

    # Construct mj model from spec.
    mj_spec = self.task.mj_spec
    mj_model = mj_spec.compile()

    def initialize_one(rng: jax.Array) -> tuple[MjxEnvState, MjxEnvState, mjx.Model, mjx.Data]:
      """Initializes one world, partitioned to cross the `jax.vmap` boundary."""
      # Construct mjx model and mjx data (non-vectorized).
      mjx_model = mjx.put_model(
        mj_model,
        impl=self.physics.impl,
      )
      mjx_data = mjx.make_data(
        mj_model,
        impl=self.physics.impl,
        naconmax=self.task.mj_model_nconmax * vectorize if self.task.mj_model_nconmax else None,
        njmax=self.task.mj_model_njmax if self.task.mj_model_njmax else None,
      )

      # Construct environment and local state (non-vectorized).
      state = MjxEnvState(
        mj_spec=mj_spec,
        mj_model=mj_model,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        env_timestep=self.task.env_timestep,
        sim_timestep=self.task.sim_timestep,
        sim_substeps=self.task.sim_substeps,
        vectorize=vectorize,
        rng=rng,
        steps=jnp.zeros((), dtype=jnp.int32),
        substeps=jnp.zeros((), dtype=jnp.int32),
        terminated=jnp.zeros((), dtype=bool),
        local=core.State(self.task, frozen=False),  # Structure can change during initialize.
      )
      state = core.entity_initialize(self.task, state, {})

      # Partition state to cross the `jax.vmap` boundary. This is needed because `mjx.Data` has
      # special vmap handlers registered with the JAX interpreter (for the 'warp' backend), which
      # are not correctly used by `eqx.filter_vmap()`. Therefore, we pass `mjx_data` separately,
      # as well as `mjx_model` for consistency and simpler out_axes specification.
      mjx_model, mjx_data = state.mjx_model, state.mjx_data
      state = state.replace(
        mjx_model=None,  # Temporarily extract.
        mjx_data=None,  # Temporarily extract.
        local=state.local.freeze(),  # Structure now frozen.
      )
      state_dynamic, state_static = eqx.partition(state, eqx.is_array)
      return state_dynamic, state_static, mjx_model, mjx_data

    # Combine dynamic state (axis 0), static state (no vec), `mjx_model` (axis 0 for special
    # fields used for model randomization/variation; broadcast other fields to dramatically reduce
    # memory usage), and `mjx_data` (axis 0 except for special fields defined in custom vmap
    # handlers registered by MJX).
    mjx_model_shape = jax.eval_shape(initialize_one, rng[0])[2]  # Get `mjx_model` obj.
    mjx_model_axes = _mjx_model_axes(mjx_model_shape, fields=self.task.mj_model_vecfields)
    state_dynamic_vec, state_static_vec, mjx_model_vec, mjx_data_vec = jax.vmap(
      initialize_one,
      in_axes=0,
      out_axes=(0, None, mjx_model_axes, 0),
    )(rng)
    state_vec = eqx.combine(state_dynamic_vec, state_static_vec)
    state_vec = state_vec.replace(mjx_model=mjx_model_vec, mjx_data=mjx_data_vec)
    return state_vec

  @T.override
  def reset(
    self,
    state: MjxEnvState,
    rng: jax.Array,
    *,
    reset: tuple[jax.Array, core.Step[MjxEnvState]] | None = None,
    subtask: Subtask | None = None,
  ) -> core.Step[MjxEnvState]:
    """Starts new episodes in all worlds, or only in the worlds selected by `reset`.

    Each reset world gets fresh `mjx_data` and local states, then `MjxTask.on_reset`, then
    `mjx_data`'s derived fields are recomputed. Its returned step has the new observation, zero
    reward, and `terminated` and `truncated` false.

    Args:
      state: The current state, `[vectorize, ...]`.
      rng: Random key of each world's new episode, `[vectorize]`.
      reset: `(mask, step)`: reset only the worlds where `mask` (`[vectorize]`, bool) is true, and
        return `step`'s values for the others. `step` is the previous step of all worlds; it cannot
        be rederived from `state`, because taking an observation is stateful. `None` resets all
        worlds.
      subtask: The subtask of each world, `[vectorize, ...]`, in `subtask_space`. `None` for a task
        without subtasks.

    Returns:
      The step of every world.
    """
    def reset_one(
      state: MjxEnvState,
      subtask: Subtask,
      rng: jax.Array,
    ) -> tuple[MjxEnvState, StepReturns]:
      """Resets one world."""
      # Reset `mjx_data` to fresh values (e.g. time, ctrl, act, qacc_warmstart, xfrc_applied) so
      # nothing carries over from the previous episode. `mjx.Data.where()` keeps the existing
      # contact/constraint buffers of the 'warp' backend, which are shared across worlds.
      mjx_data = mjx.make_data(
        state.mj_model,
        impl=self.physics.impl,
        naconmax=(
          self.task.mj_model_nconmax * state.vectorize if self.task.mj_model_nconmax else None
        ),
        njmax=self.task.mj_model_njmax if self.task.mj_model_njmax else None,
      )
      mjx_data = state.mjx_data.where(jnp.ones((), dtype=bool), mjx_data)

      # Reset the random state and local state for each episode.
      state = state.replace(
        mjx_data=mjx_data,
        rng=rng,
        steps=jnp.zeros((), dtype=jnp.int32),
        substeps=jnp.zeros((), dtype=jnp.int32),
        terminated=jnp.zeros((), dtype=bool),
      )
      state = core.entity_reset(self.task, state)

      # Task is responsible for initializing the mj/mjx state to permit flexibility. In the typical
      # case, it randomizes the `mjx_model`, then initializes the fresh `mjx_data` by invoking
      # entity methods. In more complex cases, it creates a new `mj_spec` and recompiles it to
      # `mj_model`.
      state = self.task.on_reset(state, subtask)

      # Update `mjx_data` after modifying values (e.g. qpos) to ensure all fields are fresh.
      state = state.replace(mjx_data=self.physics.forward(state))

      # Construct the step contents.
      observation, state = core.entity_observation(self.task, state)
      terminated = jnp.zeros((), dtype=bool)
      truncated = jnp.zeros((), dtype=bool)
      rewards = self.task.get_rewards(state, terminated)
      rewards = { k: (w, jnp.zeros_like(v)) for k, (w, v) in rewards.items() }
      reward = jnp.zeros(())
      metrics = self.task.get_metrics(state, terminated, rewards)
      return state, (observation, reward, terminated, truncated, metrics)

    def reset_one_where(
      state: MjxEnvState,
      subtask: Subtask,
      rng: jax.Array,
      mask: jax.Array,
      reset_returns: StepReturns,
    ) -> tuple[MjxEnvState, StepReturns]:
      """Resets one world if `mask`, and otherwise passes its state and `reset_returns` through."""
      # `jax.lax.cond` with a batched predicate evaluates both branches under `jax.vmap` anyway,
      # and fails for the 'warp' backend since `mjx.forward()` has a custom vmap rule. Instead,
      # always reset and then merge.
      # `reset_one` mutates `state.local` via `core.State.set()`, which invalidates the instance
      # it was called on; clone it first so the original `state` isn't left stale for merging.
      reset_state, returns = reset_one(state.replace(local=state.local.clone()), subtask, rng)
      # `mjx.Data.where()` keeps the contact/constraint buffers of the 'warp' backend, which are
      # shared across worlds. Leaves untouched by the reset (e.g. non-vectorized `mjx_model`
      # fields) are passed through so that they remain unbatched.
      mjx_data = state.mjx_data.where(mask, reset_state.mjx_data)
      merge = lambda new, old: old if new is old else jnp.where(mask, new, old)
      state = jax.tree.map(merge, reset_state.replace(mjx_data=None), state.replace(mjx_data=None))
      returns = jax.tree.map(merge, returns, reset_returns)
      return state.replace(mjx_data=mjx_data), returns

    # When `reset` is not specified, we reset all environments.
    # When `reset` is specified, we mask the reset with `mask` to reset only the environments that
    # need to be reset, and pass through the previous step for non-reset environments.
    subtask = {} if subtask is None else subtask
    self.subtask_space(state).validate(subtask)
    if reset is None:
      state, (observation, reward, terminated, truncated, metrics) = vmap_state_update(
        reset_one,
        mj_model_vecfields=self.task.mj_model_vecfields,
        in_axes=(0, 0),
      )(state, subtask, rng)
    else:
      mask, step = reset
      state, (observation, reward, terminated, truncated, metrics) = vmap_state_update(
        reset_one_where,
        mj_model_vecfields=self.task.mj_model_vecfields,
        in_axes=(0, 0, 0, 0),
      )(
        state,
        subtask,
        rng,
        mask,
        (step.observation, step.reward, step.terminated, step.truncated, step.metrics),
      )

    return core.Step(
      state=state,
      observation=observation,
      reward=reward,
      terminated=terminated,
      truncated=truncated,
      metrics=metrics,
    )

  @T.override
  def step(self, state: MjxEnvState, action: core.Action) -> core.Step[MjxEnvState]:
    """Advances every world by one environment step of `sim_substeps` simulation substeps.

    The task's hooks run in the order listed in `MjxTask`. `terminated` is sticky: once true, it
    stays true for the rest of the episode, and later steps have zero reward. `truncated` is true
    from step `truncate` on.

    Args:
      state: The current state, `[vectorize, ...]`.
      action: Action of each world, `[vectorize, ...]`, in `action_space`.

    Returns:
      The step of every world.
    """
    def step_one(state: MjxEnvState, action: core.Action) -> tuple[MjxEnvState, StepReturns]:
      """Runs one environment step of one world: `sim_substeps` simulation substeps."""
      # Increment steps at start, so that task/entities can use the step number.
      state = state.replace(steps=state.steps + 1)
      state = self.task.on_step_start(state, action)
      state_dynamic, state_static = eqx.partition(state, eqx.is_array)

      def substep_one(state: MjxEnvState) -> MjxEnvState:
        """Runs one simulation substep of one world."""
        # Increment substeps at start, so that task/entities can use the substep number.
        state = state.replace(substeps=state.substeps + 1)
        state = self.task.on_substep_start(state, action)
        state = core.entity_action(self.task, state, action)
        state = state.replace(mjx_data=self.physics.step(state))
        state = self.task.on_substep_end(state, action)
        return state

      def substep_and_observation_one(
        i: int,
        state_dynamic: MjxEnvState,
      ) -> MjxEnvState:
        """Runs one simulation substep and an intermediate observation, on the dynamic state."""
        state = eqx.combine(state_dynamic, state_static)
        state = substep_one(state)
        _, state = core.entity_observation(self.task, state)  # Discard intermediate observations.
        state_dynamic, _ = eqx.partition(state, eqx.is_array)
        return state_dynamic

      # Run all substeps except last using `jax.lax.fori_loop()`, which is more time-efficient than
      # a for-loop due to faster compilation, and is more memory-efficient than `jax.lax.scan`
      # that would store all intermediate states in memory and stack them in the final output.
      # `jax.lax.fori_loop()` only works on dynamic states.
      state_dynamic = jax.lax.fori_loop(
        0, self.task.sim_substeps - 1, substep_and_observation_one, state_dynamic
      )
      # Run the last substep manually to enable `task.on_substep_end()` to possibly modify the
      # state before rendering and constructing the step.
      state = eqx.combine(state_dynamic, state_static)
      state = substep_one(state)
      state = self.task.on_step_end(state, action)

      # Get the `observation`/`terminated`/`reward` from the final state of the environment step.
      # Once `terminated = True` in an episode, all subsequent `terminated = True` and `reward = 0`.
      # Once `truncated = True` in an episode all subsequent `truncated = True` due to >= check.
      observation, state = core.entity_observation(self.task, state)
      terminated = self.task.get_terminated(state).astype(bool) | state.terminated
      truncated = (
        jnp.zeros((), dtype=bool) if self.truncate is None else state.steps >= self.truncate
      )
      rewards = self.task.get_rewards(state, terminated)
      reward = self.task.get_reward(rewards) * (~state.terminated).astype(float)
      metrics = self.task.get_metrics(state, terminated, rewards)

      state = state.replace(terminated=terminated)
      return state, (observation, reward, terminated, truncated, metrics)

    state, (observation, reward, terminated, truncated, metrics) = vmap_state_update(
      step_one,
      mj_model_vecfields=self.task.mj_model_vecfields,
      in_axes=(0,),
    )(state, action)

    return core.Step(
      state=state,
      observation=observation,
      reward=reward,
      terminated=terminated,
      truncated=truncated,
      metrics=metrics,
    )

  def subtask_space(self, state: MjxEnvState) -> SubtaskSpace:
    """Returns the space of the `subtask` passed to `reset`, vectorized over the worlds."""
    subtask_space = vmap_state(
      lambda state: self.task.subtask_space(state),
      mj_model_vecfields=self.task.mj_model_vecfields,
    )(state)
    subtask_space = subtask_space.vectorize(state.vectorize)
    return subtask_space

  @T.override
  def action_space(self, state: MjxEnvState) -> core.ActionSpace:
    """Returns the space of the actions passed to `step`, vectorized over the worlds."""
    action_space = vmap_state(
      lambda state: core.entity_action_space(self.task, state),
      mj_model_vecfields=self.task.mj_model_vecfields,
    )(state)
    action_space = action_space.vectorize(state.vectorize)
    return T.cast(core.ActionSpace, action_space)

  @T.override
  def observation_space(self, state: MjxEnvState) -> core.ObservationSpace:
    """Returns the space of the observations in each step, vectorized over the worlds."""
    observation_space = vmap_state(
      lambda state: core.entity_observation_space(self.task, state),
      mj_model_vecfields=self.task.mj_model_vecfields,
    )(state)
    observation_space = observation_space.vectorize(state.vectorize)
    return T.cast(core.ObservationSpace, observation_space)


# ==================================================================================================
# Rendering

RenderLabel = T.Literal[
  'none', 'body', 'joint', 'geom', 'site', 'camera', 'light', 'tendon', 'actuator', 'constraint',
  'flex', 'skin', 'contactpoint', 'contactforce', 'island',
]
"""Names of `mj.mjtLabel` values (without the `mjLABEL_` prefix)."""

RenderFrame = T.Literal['none', 'body', 'geom', 'site', 'camera', 'light', 'contact', 'world']
"""Names of `mj.mjtFrame` values (without the `mjFRAME_` prefix)."""

RenderVisFlag = T.Literal[
  'convexhull', 'texture', 'joint', 'camera', 'actuator', 'activation', 'light', 'tendon',
  'rangefinder', 'constraint', 'inertia', 'sclinertia', 'pertforce', 'pertobj', 'contactpoint',
  'island', 'contactforce', 'contactsplit', 'transparent', 'autoconnect', 'com', 'select',
  'static', 'skin', 'flexvert', 'flexedge', 'flexface', 'flexskin', 'bodybvh', 'meshbvh',
  'sdfiter',
]
"""Names of `mj.mjtVisFlag` values (without the `mjVIS_` prefix)."""

RenderRndFlag = T.Literal[
  'shadow', 'wireframe', 'reflection', 'additive', 'skybox', 'fog', 'haze', 'segment', 'idcolor',
  'cull_face',
]
"""Names of `mj.mjtRndFlag` values (without the `mjRND_` prefix)."""

RenderBackend = T.Literal['mujoco', 'warp']
"""Renderer of `render`: MuJoCo's OpenGL renderer (CPU or GPU) or MJX's Warp ray tracer (GPU)."""

_RENDER_STATE_FIELDS = ('qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'xfrc_applied')
"""`mjx_data` fields that determine a rendered frame (after `mj_forward`)."""


class _RenderTaskKwargs(T.TypedDict):
  """Keyword arguments of `_render_tasks` shared by every task (see `_render_mujoco`)."""
  camera_ids: list[int]
  indices: list[int]
  width: int
  height: int
  depth: bool
  segmentation: bool
  scene_modify: T.Callable[[mj.MjvScene], None] | None
  rnd: dict[str, bool]


class MjxRenderKwargs(T.TypedDict, total=False):
  """Keyword arguments of `render`."""
  index: int | T.Sequence[int]
  size: tuple[int, int]
  camera: str | int | T.Sequence[str | int]
  depth: bool
  segmentation: bool
  scene_option: mj.MjvOption | None
  scene_modify: T.Callable[[mj.MjvScene], None] | None
  geomgroup: T.Sequence[int]
  sitegroup: T.Sequence[int]
  jointgroup: T.Sequence[int]
  tendongroup: T.Sequence[int]
  actuatorgroup: T.Sequence[int]
  flexgroup: T.Sequence[int]
  skingroup: T.Sequence[int]
  label: mj.mjtLabel | RenderLabel
  frame: mj.mjtFrame | RenderFrame
  vis: T.Mapping[RenderVisFlag, bool]
  rnd: T.Mapping[RenderRndFlag, bool]
  backend: RenderBackend
  workers: int | None
  verbose: bool


def render(
  states: T.Sequence[MjxEnvState],
  *,
  index: int | T.Sequence[int] = 0,
  size: tuple[int, int] = (480, 640),
  camera: str | int | T.Sequence[str | int] = 0,
  depth: bool = False,
  segmentation: bool = False,
  scene_option: mj.MjvOption | None = None,
  scene_modify: T.Callable[[mj.MjvScene], None] | None = None,
  geomgroup: T.Sequence[int] = (0, 1, 2),
  sitegroup: T.Sequence[int] = (0, 1, 2),
  jointgroup: T.Sequence[int] = (0, 1, 2),
  tendongroup: T.Sequence[int] = (0, 1, 2),
  actuatorgroup: T.Sequence[int] = (0, 1, 2),
  flexgroup: T.Sequence[int] = (0, 1, 2),
  skingroup: T.Sequence[int] = (0, 1, 2),
  label: mj.mjtLabel | RenderLabel = 'none',
  frame: mj.mjtFrame | RenderFrame = 'none',
  vis: T.Mapping[RenderVisFlag, bool] = types.MappingProxyType({}),
  rnd: T.Mapping[RenderRndFlag, bool] = types.MappingProxyType({}),
  backend: RenderBackend = 'mujoco',
  workers: int | None = None,
  verbose: bool = True,
) -> list[np.ndarray]:
  """Renders worlds of a sequence of states (for example, a rollout) to images.

  Args:
    states: States of one environment, sharing one `mj_model`.
    index: The world to render, or a sequence of worlds.
    size: Image `(height, width)` in pixels.
    camera: The camera (name or id) to render from, or a sequence of cameras.
    depth: Whether to render depth instead of color.
    segmentation: Whether to render segmentation instead of color.
    scene_option: MuJoCo visualization options. If given, the `*group`, `label`, `frame` and `vis`
      arguments are ignored. 'mujoco' renderer only.
    scene_modify: Function that edits the scene before each frame is drawn (e.g. to add markers).
      'mujoco' renderer only.
    geomgroup: Visible geom groups.
    sitegroup: Visible site groups ('mujoco' renderer only, as are the other groups below).
    jointgroup: Visible joint groups.
    tendongroup: Visible tendon groups.
    actuatorgroup: Visible actuator groups.
    flexgroup: Visible flex groups.
    skingroup: Visible skin groups.
    label: What to label ('mujoco' renderer only).
    frame: Which frames to draw ('mujoco' renderer only).
    vis: Visualization flags to set.
    rnd: Rendering flags to set.
    backend: The renderer (see `RenderBackend`). The Warp renderer supports only some flags, and
      warns about the others if `verbose`.
    workers: Number of Ray worker processes to render in parallel ('mujoco' renderer only), or
      `None` to render in this process.
    verbose: Whether to warn about arguments the renderer does not support.

  Returns:
    One image array per state: `[height, width, ...]`, with leading `[camera]` and `[index]` axes
    (in that order) when `camera` or `index` is a sequence. Color images are `uint8` `[..., 3]`;
    depth and segmentation images have the renderer's own dtype and channels.

  Raises:
    ValueError: If the states have different models, a camera is unknown, both `depth` and
      `segmentation` are set, or `workers` is invalid.
  """
  if len(states) == 0: return []
  mj_model = states[0].mj_model
  for i, state in enumerate(states):
    if state.mj_model != mj_model: raise ValueError(f'Different `mj_model` for state {i}')

  # Validate the arguments.
  if workers is not None:
    if workers <= 0:
      raise ValueError('Number of workers must be positive.')
    if backend != 'mujoco':
      raise ValueError('Multithreading only supported for mujoco backend.')
    cpus = len(os.sched_getaffinity(0))
    if workers > cpus:
      raise ValueError(f'Number of workers {workers} must be <= #CPUs: {cpus}.')
  if depth and segmentation:
    raise ValueError('Cannot render both depth and segmentation simultaneously')

  # Normalize the worlds and cameras to sequences, and cameras to ids.
  index_singleton = isinstance(index, int)
  camera_singleton = isinstance(camera, (str, int))
  indices = [index] if isinstance(index, int) else list(index)
  cameras = [camera] if isinstance(camera, (str, int)) else list(camera)
  try:
    camera_ids = [mj_model.camera(c).id for c in cameras]
  except KeyError:
    valid = [mj_model.camera(i).name for i in range(mj_model.ncam)]
    raise ValueError(f'Invalid camera: {camera!r}. Valid cameras: {valid}') from None

  # Render every (state, camera, world): [time, camera, index, height, width, ...].
  match backend:
    case 'mujoco':
      if scene_option is None:
        scene_option = _make_scene_option(
          geomgroup=geomgroup,
          sitegroup=sitegroup,
          jointgroup=jointgroup,
          tendongroup=tendongroup,
          actuatorgroup=actuatorgroup,
          flexgroup=flexgroup,
          skingroup=skingroup,
          label=label,
          frame=frame,
          vis=vis,
        )
      images = _render_mujoco(
        states,
        camera_ids,
        indices,
        size=size,
        depth=depth,
        segmentation=segmentation,
        scene_option=scene_option,
        scene_modify=scene_modify,
        rnd=rnd,
        workers=workers,
      )
    case 'warp':
      if verbose:
        for name, value in dict(
          scene_option=scene_option,
          scene_modify=scene_modify,
          label=label,
          frame=frame,
        ).items():
          if value is not None and value != 'none':
            warnings.warn(f'Warp renderer does not support {name}={value!r}.')
      images = _render_warp(
        states,
        camera_ids,
        indices,
        size=size,
        depth=depth,
        segmentation=segmentation,
        geomgroup=geomgroup,
        vis=vis,
        rnd=rnd,
        verbose=verbose,
      )
    case _:
      raise ValueError(f'Unknown render backend: {backend!r}.')

  # Drop the camera and world axes that were not requested as sequences.
  if camera_singleton:
    images = images[:, 0]
    if index_singleton:
      images = images[:, 0]
  elif index_singleton:
    images = images[:, :, 0]
  return list(images)


def _enum_value(
  enum: type[mj.mjtLabel] | type[mj.mjtFrame] | type[mj.mjtVisFlag] | type[mj.mjtRndFlag],
  prefix: str,
  kind: str,
  name: str,
) -> int:
  """Returns the value of the MuJoCo enum member `<prefix><NAME>`.

  Raises:
    ValueError: If there is no such member; lists the valid names.
  """
  try:
    return getattr(enum, f'{prefix}{name.upper()}').value
  except AttributeError:
    valid = [k.replace(prefix, '').lower() for k in enum.__members__ if k.startswith(prefix)]
    raise ValueError(f'Invalid {kind}: {name!r}. Valid {kind}: {valid}') from None


def _make_scene_option(
  *,
  geomgroup: T.Sequence[int],
  sitegroup: T.Sequence[int],
  jointgroup: T.Sequence[int],
  tendongroup: T.Sequence[int],
  actuatorgroup: T.Sequence[int],
  flexgroup: T.Sequence[int],
  skingroup: T.Sequence[int],
  label: mj.mjtLabel | RenderLabel,
  frame: mj.mjtFrame | RenderFrame,
  vis: T.Mapping[RenderVisFlag, bool],
) -> mj.MjvOption:
  """Returns MuJoCo visualization options with the given visible groups, labels, frames and flags
  (arguments as in `render`)."""
  scene_option = mj.MjvOption()

  # Visible groups.
  for option_groups, groups in (
    (scene_option.geomgroup, geomgroup),
    (scene_option.sitegroup, sitegroup),
    (scene_option.jointgroup, jointgroup),
    (scene_option.tendongroup, tendongroup),
    (scene_option.actuatorgroup, actuatorgroup),
    (scene_option.flexgroup, flexgroup),
    (scene_option.skingroup, skingroup),
  ):
    option_groups[:] = 0
    option_groups[list(groups)] = 1

  # Labels, frames and visualization flags.
  scene_option.label = (
    label.value if isinstance(label, mj.mjtLabel)
    else _enum_value(mj.mjtLabel, 'mjLABEL_', 'label', label)
  )
  scene_option.frame = (
    frame.value if isinstance(frame, mj.mjtFrame)
    else _enum_value(mj.mjtFrame, 'mjFRAME_', 'frame', frame)
  )
  for name, value in vis.items():
    scene_option.flags[_enum_value(mj.mjtVisFlag, 'mjVIS_', 'vis flag', name)] = bool(value)
  return scene_option


def _render_mujoco(
  states: T.Sequence[MjxEnvState],
  camera_ids: list[int],
  indices: list[int],
  *,
  size: tuple[int, int],
  depth: bool,
  segmentation: bool,
  scene_option: mj.MjvOption,
  scene_modify: T.Callable[[mj.MjvScene], None] | None,
  rnd: T.Mapping[RenderRndFlag, bool],
  workers: int | None,
) -> np.ndarray:
  """Renders with MuJoCo's renderer, here or in Ray workers. Returns `[time, camera, index, ...]`
  images."""
  mj_model = states[0].mj_model
  height, width = size

  # The fields that determine each frame, as NumPy arrays (states cannot be pickled for workers).
  frames = [
    {name: np.asarray(getattr(state.mjx_data, name)) for name in _RENDER_STATE_FIELDS}
    for state in states
  ]

  # One task per (state, world, camera), as positions in `states`, `indices` and `camera_ids`,
  # ordered so that each world's state is loaded once for all its cameras.
  tasks = [
    (t, c, i)
    for t in range(len(states)) for i in range(len(indices)) for c in range(len(camera_ids))
  ]
  render_kwargs = _RenderTaskKwargs(
    camera_ids=camera_ids,
    indices=indices,
    width=width,
    height=height,
    depth=depth,
    segmentation=segmentation,
    scene_modify=scene_modify,
    rnd={name: value for name, value in rnd.items()},
  )

  # Render here, or split the tasks between Ray workers that share the frames.
  if workers is None:
    results = _render_tasks(mj_model, frames, tasks, scene_option=scene_option, **render_kwargs)
  else:
    import ray  # Only needed to render in parallel.
    # Ray's type stubs only accept functions with positional-only arguments.
    render_tasks_remote = ray.remote(_render_tasks_in_worker)  # type: ignore
    frames_ref = ray.put(frames)
    futures = [
      render_tasks_remote.remote(
        mj_model,
        frames_ref,
        [(int(t), int(c), int(i)) for t, c, i in chunk],
        scene_option=serialize_scene_option(scene_option),
        **render_kwargs,
      )
      for chunk in np.array_split(np.array(tasks), workers) if len(chunk) > 0
    ]
    results = [result for future in futures for result in ray.get(future)]

  # Assemble the images. Allocate from the first image, since color, depth and segmentation images
  # differ in dtype and channels.
  first_image = results[0][1]
  images = np.empty(
    (len(states), len(camera_ids), len(indices)) + first_image.shape,
    first_image.dtype,
  )
  for (t, c, i), image in results:
    images[t, c, i] = image
  return images


def _render_tasks(
  mj_model: mj.MjModel,
  frames: T.Sequence[dict[str, np.ndarray]],
  tasks: T.Sequence[tuple[int, int, int]],
  *,
  camera_ids: list[int],
  indices: list[int],
  width: int,
  height: int,
  depth: bool,
  segmentation: bool,
  scene_option: mj.MjvOption,
  scene_modify: T.Callable[[mj.MjvScene], None] | None,
  rnd: dict[str, bool],
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
  """Renders each `(state, camera, world)` task with one MuJoCo renderer. Returns each task with
  its image."""
  with mj.Renderer(mj_model, width=width, height=height) as renderer:
    renderer = T.cast(mj.Renderer, renderer)
    if depth: renderer.enable_depth_rendering()
    if segmentation: renderer.enable_segmentation_rendering()
    for name, value in rnd.items():
      renderer.scene.flags[_enum_value(mj.mjtRndFlag, 'mjRND_', 'rnd flag', name)] = bool(value)

    mj_data = mj.MjData(mj_model)
    loaded = None  # The (state, world) loaded into `mj_data`.
    results = []
    for t, c, i in tasks:
      # Load the world's state, and recompute its derived fields, once for all its cameras.
      if loaded != (t, i):
        for name, values in frames[t].items():
          getattr(mj_data, name)[:] = values[indices[i]]
        mj.mj_forward(mj_model, mj_data)
        loaded = (t, i)

      renderer.update_scene(mj_data, camera=camera_ids[c], scene_option=scene_option)
      if scene_modify is not None: scene_modify(renderer.scene)
      results.append(((t, c, i), renderer.render()))
  return results


def _render_tasks_in_worker(
  mj_model: mj.MjModel,
  frames: T.Sequence[dict[str, np.ndarray]],
  tasks: T.Sequence[tuple[int, int, int]],
  *,
  scene_option: dict[str, T.Any],
  **kwargs: T.Unpack[_RenderTaskKwargs],
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
  """`_render_tasks` in a Ray worker process, with its own OpenGL context and the scene option
  serialized by `serialize_scene_option`."""
  context = mj.GLContext(kwargs['width'], kwargs['height'])
  context.make_current()
  try:
    return _render_tasks(
      mj_model,
      frames,
      tasks,
      scene_option=deserialize_scene_option(scene_option),
      **kwargs,
    )
  finally:
    context.free()


def _render_warp(
  states: T.Sequence[MjxEnvState],
  camera_ids: list[int],
  indices: list[int],
  *,
  size: tuple[int, int],
  depth: bool,
  segmentation: bool,
  geomgroup: T.Sequence[int],
  vis: T.Mapping[RenderVisFlag, bool],
  rnd: T.Mapping[RenderRndFlag, bool],
  verbose: bool,
) -> np.ndarray:
  """Renders with MJX's Warp ray tracer. Returns `[time, camera, index, ...]` images."""
  mj_model = states[0].mj_model

  # The render context options that correspond to MuJoCo's flags: option -> (flag group, flag).
  flags: dict[str, tuple[str, str]] = dict(
    use_textures=('vis', 'texture'),
    use_shadows=('rnd', 'shadow'),
    flex_render_smooth=('vis', 'flexskin'),
    render_skybox=('rnd', 'skybox'),
    enable_backface_culling=('rnd', 'cull_face'),
  )
  flag_groups: dict[str, dict[str, bool]] = {
    'vis': {name: value for name, value in vis.items()},
    'rnd': {name: value for name, value in rnd.items()},
  }
  if verbose:
    for group, values in flag_groups.items():
      supported = [flag for flag_group, flag in flags.values() if flag_group == group]
      for flag in values:
        if flag not in supported:
          warnings.warn(f'Warp renderer does not support {group} flag {flag!r}.')

  # Options forwarded to the render context's constructor.
  options: dict[str, T.Any] = dict(
    cam_res=(size[1], size[0]),  # (width, height) in MJX vs (height, width) in MuJoCo.
    render_rgb=True,
    render_depth=depth,
    render_seg=segmentation,
    enabled_geom_groups=geomgroup,
    cam_active=[i in camera_ids for i in range(mj_model.ncam)],
    use_precomputed_rays=True,
  ) | {
    option: flag_groups[group][flag]
    for option, (group, flag) in flags.items() if flag in flag_groups[group]
  }
  render_context = mjx.create_render_context(mjm=mj_model, nworld=len(indices), **options)

  @jax.jit
  def render_worlds(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    render_context_pytree: mjxrc.RenderContextPytree,
  ) -> jax.Array:
    """Renders every camera of the selected worlds: `[camera, index, ...]`."""
    # Render the packed buffers, then unpack each camera's image from the requested one.
    mjx_data = mjx.refit_bvh(mjx_model, mjx_data, render_context_pytree)
    if segmentation:
      _, _, packed, _ = mjx.render_with_segmentation(mjx_model, mjx_data, render_context_pytree)
      get_image = mjx.get_segmentation
    else:
      rgb, depth_packed, _ = mjx.render(mjx_model, mjx_data, render_context_pytree)
      packed = depth_packed if depth else rgb
      get_image = (
        functools.partial(mjx.get_depth, depth_scale=1.0)
        if depth else mjx.get_rgb
      )
    return jnp.stack([
      get_image(render_context_pytree, camera_id, packed) for camera_id in camera_ids
    ])

  # Per-world model fields have a leading world axis that the shared ones lack (compared against an
  # unbatched model); select the rendered worlds from both the model and the data.
  world_index = jnp.asarray(indices)
  shared_model = mjx.put_model(mj_model, impl='warp')
  render_context_pytree = render_context.pytree()
  images = []
  for state in states:
    mjx_model = jax.tree.map(
      lambda x, shared: x[world_index] if eqx.is_array(x) and x.shape != shared.shape else x,
      state.mjx_model,
      shared_model,
    )
    mjx_data = jax.tree.map(lambda x: x[world_index], state.mjx_data)
    images.append(np.asarray(render_worlds(mjx_model, mjx_data, render_context_pytree)))
  return np.stack(images)


def serialize_scene_option(scene_option: mj.MjvOption) -> dict[str, T.Any]:
  """Serializes a `mj.MjvOption` to a dictionary, which can be pickled."""
  return {
    'geomgroup': scene_option.geomgroup.tolist(),
    'sitegroup': scene_option.sitegroup.tolist(),
    'jointgroup': scene_option.jointgroup.tolist(),
    'tendongroup': scene_option.tendongroup.tolist(),
    'actuatorgroup': scene_option.actuatorgroup.tolist(),
    'flexgroup': scene_option.flexgroup.tolist(),
    'skingroup': scene_option.skingroup.tolist(),
    'label': scene_option.label,
    'frame': scene_option.frame,
    'flags': scene_option.flags.tolist(),
  }


def deserialize_scene_option(data: dict[str, T.Any]) -> mj.MjvOption:
  """Deserializes a dictionary to a `mj.MjvOption`."""
  scene_option = mj.MjvOption()
  scene_option.geomgroup[:] = data['geomgroup']
  scene_option.sitegroup[:] = data['sitegroup']
  scene_option.jointgroup[:] = data['jointgroup']
  scene_option.tendongroup[:] = data['tendongroup']
  scene_option.actuatorgroup[:] = data['actuatorgroup']
  scene_option.flexgroup[:] = data['flexgroup']
  scene_option.skingroup[:] = data['skingroup']
  scene_option.label = mj.mjtLabel(data['label'])
  scene_option.frame = mj.mjtFrame(data['frame'])
  scene_option.flags[:] = data['flags']
  return scene_option


# ==================================================================================================
# Contact forces


def _contact_force_world_jax(
  model: mjx.Model,
  data: mjx.Data,
) -> tuple[jax.Array, jax.Array]:
  """Returns the linear contact force of each contact in world frame, `[ncon, 3]`, and the contact
  geom pairs, `[ncon, 2]`, for the 'jax' data layout (also used by the 'mujoco' backend).

  This is `mjx._src.support.contact_force` (MuJoCo's `mj_contactForce`) for all contacts at once.
  Each contact's constraint rows are contiguous from its `efc_address`. Contacts without rows
  (inactive, or beyond their margin) have zero force.
  """
  # `data._impl` is `mjx.DataJAX`, or a subset of it loaded by `utils.rollout`.
  impl = T.cast(mjx.DataJAX, data._impl)
  efc_force = impl.efc_force  # [nefc]
  efc_address = jnp.asarray(impl.contact.efc_address)  # [ncon]
  condim = jnp.asarray(impl.contact.dim)  # [ncon]
  nefc = efc_force.shape[0]

  def get_efc_force(address: jax.Array) -> jax.Array:
    """Constraint force of each row address, zero for addresses outside the rows."""
    valid = (address >= 0) & (address < nefc)
    return jnp.where(valid, efc_force[jnp.clip(address, 0, max(nefc - 1, 0))], 0.)

  # Linear force in the contact frame: [normal, tangent1, tangent2].
  if model.opt.cone == mjx.ConeType.PYRAMIDAL:
    # A pyramidal cone has 2 rows per tangent direction (condim <= 6): the normal force is the sum
    # of all rows, each tangent force the difference of its pair times the friction coefficient.
    edges = jnp.arange(5)
    is_edge = edges < condim[:, None] - 1  # [ncon, 5]
    edge1 = get_efc_force(efc_address[:, None] + 2 * edges)
    edge2 = get_efc_force(efc_address[:, None] + 2 * edges + 1)
    normal = jnp.where(
      condim == 1,
      get_efc_force(efc_address),
      jnp.where(is_edge, edge1 + edge2, 0.).sum(axis=1),
    )  # [ncon]
    tangent = jnp.where(
      is_edge[:, :2],
      (edge1 - edge2)[:, :2] * impl.contact.friction[:, :2],
      0.,
    )  # [ncon, 2]
    force = jnp.concatenate([normal[:, None], tangent], axis=1)  # [ncon, 3]
  elif model.opt.cone == mjx.ConeType.ELLIPTIC:
    # An elliptic cone has one row per force component.
    dims = jnp.arange(3)
    force = jnp.where(
      dims < condim[:, None],
      get_efc_force(efc_address[:, None] + dims),
      0.,
    )  # [ncon, 3]
  else:
    raise ValueError(f'Unknown cone type: {model.opt.cone}')

  # Rotate to world frame (the frame's rows are the normal and tangent axes).
  force = jnp.einsum('ci,cij->cj', force, impl.contact.frame)  # [ncon, 3]
  return jnp.where((efc_address >= 0)[:, None], force, 0.), impl.contact.geom


def _contact_force_world_warp(
  model: mjx.Model,
  data: mjx.Data,
) -> tuple[jax.Array, jax.Array]:
  """Returns the linear contact force of each contact in world frame, `[naconmax, 3]`, and the
  contact geom pairs, `[naconmax, 2]`, for the 'warp' backend. Contacts of other worlds and
  inactive contacts have zero force.

  Unlike the 'jax' backend, the 'warp' backend keeps the contacts of all worlds in one buffer that
  is not batched by `jax.vmap` (see `mjx.warp.types.DATA_NON_VMAP`), tagged by world, while
  constraint forces are batched per world. Thus, this must be called within `vmap_state` or
  `vmap_state_update` to know which world's contacts to select.

  Raises:
    ValueError: If called outside `vmap_state` or `vmap_state_update`.
  """
  # `data._impl` is `mjx.warp.types.DataWarp`, or a subset of it loaded by `utils.rollout`.
  impl = T.cast(mjx_warp_types.DataWarp, data._impl)
  try:
    world_id = jax.lax.axis_index(WORLD_AXIS)
  except NameError as e:
    raise ValueError(
      "Contact forces with the 'warp' backend must be computed within `vmap_state` or "
      "`vmap_state_update`, since contacts of all worlds share one unbatched buffer."
    ) from e
  efc_force = impl.efc__force  # [njmax]
  efc_address = impl.contact__efc_address  # [naconmax, nmaxpyramid]
  condim = impl.contact__dim  # [naconmax]
  njmax = efc_force.shape[0]
  def get_efc_force(address: jax.Array) -> jax.Array:
    """Constraint force of each row address, zero for unused addresses (>= `njmax`)."""
    return jnp.where(address < njmax, efc_force[jnp.clip(address, 0, njmax - 1)], 0.)

  # Extract linear force for each contact in contact frame, as in MuJoCo Warp's `contact_force_fn`:
  # https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/support.py
  if model.opt.cone == mjx.ConeType.PYRAMIDAL:
    # Decode pyramid edges [naconmax, 5] for condim <= 6.
    address = efc_address[:, 0]
    edges = jnp.arange(5)
    is_edge = edges < condim[:, None] - 1
    edge1 = get_efc_force(address[:, None] + 2 * edges)
    edge2 = get_efc_force(address[:, None] + 2 * edges + 1)
    normal = jnp.where(
      condim == 1,
      get_efc_force(address),
      jnp.where(is_edge, edge1 + edge2, 0.).sum(axis=1),
    )  # [naconmax]
    tangent = jnp.where(
      is_edge[:, :2], (edge1 - edge2)[:, :2] * impl.contact__friction[:, :2], 0.
    )  # [naconmax, 2]
    force = jnp.concatenate([normal[:, None], tangent], axis=1)  # [naconmax, 3]
  elif model.opt.cone == mjx.ConeType.ELLIPTIC:
    dims = min(3, efc_address.shape[1])
    force = jnp.where(
      jnp.arange(dims) < condim[:, None], get_efc_force(efc_address[:, :dims]), 0.
    )
    force = jnp.pad(force, ((0, 0), (0, 3 - dims)))  # [naconmax, 3]
  else:
    raise ValueError(f'Unknown cone type: {model.opt.cone}')

  # Rotate to world frame and zero out inactive contacts and contacts of other worlds.
  force = jnp.einsum('ci,cij->cj', force, impl.contact__frame)  # [naconmax, 3]
  is_valid = (
    (jnp.arange(force.shape[0]) < impl.nacon[0])
    & (impl.contact__worldid == world_id)
    & (efc_address[:, 0] >= 0)
  )  # [naconmax]
  return jnp.where(is_valid[:, None], force, 0.), impl.contact__geom


def contact_force_geom_world(
  model: mjx.Model,
  data: mjx.Data,
  geom_ids: jax.Array | T.Sequence[int],
) -> jax.Array:
  """Returns the net linear contact force on each geom, in world frame.

  Args:
    model: MJX model.
    data: MJX data of one world. With the 'warp' backend, call this within `vmap_state` or
      `vmap_state_update`.
    geom_ids: The geoms, `[ngeom]`.

  Returns:
    The sum of the contact forces exerted *on* each geom by the geoms it touches, `[ngeom, 3]`
    (for a body resting on the floor, about its weight, upward).
  """
  if model.impl == mjx.Impl.WARP:
    contact_forces, contact_geoms = _contact_force_world_warp(model, data)  # [n_contacts, 3|2]
  else:
    contact_forces, contact_geoms = _contact_force_world_jax(model, data)  # [n_contacts, 3|2]
  geom1 = contact_geoms[:, 0]  # [n_contacts,]
  geom2 = contact_geoms[:, 1]  # [n_contacts,]

  # For each geom, find all contacts involving that geom and sum the contact forces.
  geom_ids = jnp.asarray(geom_ids)
  matches_geom1 = geom1[None, :] == geom_ids[:, None]  # [n_geoms, n_contacts]
  matches_geom2 = geom2[None, :] == geom_ids[:, None]  # [n_geoms, n_contacts]
  sign = jnp.where(matches_geom1, -1.0, 1.0)  # [n_geoms, n_contacts]
  masked = jnp.where(
    (matches_geom1 | matches_geom2)[:, :, None],
    sign[..., None] * contact_forces[None, :, :],
    0.0,
  )  # [n_geoms, n_contacts, 3]
  return masked.sum(axis=1)  # [n_geoms, 3]

def contact_force_body_world(
  model: mjx.Model,
  data: mjx.Data,
  body_ids: jax.Array | T.Sequence[int],
) -> jax.Array:
  """Returns the net linear contact force on each body (summed over its geoms), in world frame.

  Args:
    model: MJX model.
    data: MJX data of one world. With the 'warp' backend, call this within `vmap_state` or
      `vmap_state_update`.
    body_ids: The bodies, `[nbody]`.

  Returns:
    The contact forces exerted *on* each body, `[nbody, 3]` (see `contact_force_geom_world`).
  """
  # Sum the net contact forces on geoms per body, in world frame.
  return jnp.stack([
    contact_force_geom_world(
      model,
      data,
      model.body_geomadr[body_id] + jnp.arange(model.body_geomnum[body_id])
    ).sum(axis=0) for body_id in body_ids
  ])  # [n_bodies, 3]
