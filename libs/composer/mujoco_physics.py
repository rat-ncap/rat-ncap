import dataclasses
import functools
import typing as T
import warnings

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.mjx._src.collision_driver as mjx_collision_driver
import mujoco.mjx._src.constraint as mjx_constraint
import mujoco.mjx._src.types as mjx_types
import numpy as np
import scipy.linalg

__all__ = ['MujocoPhysics', 'get_physics']

# ==================================================================================================
# Fields

STATE_FIELDS = (
  'time', 'qpos', 'qvel', 'act', 'history', 'qacc_warmstart', 'plugin_state', 'ctrl',
  'qfrc_applied', 'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'userdata',
)
"""MuJoCo's integration state (`mjSTATE_INTEGRATION`): everything the next step depends on."""

CONTACT_FIELDS = (
  'dist', 'pos', 'frame', 'includemargin', 'friction', 'solref', 'solreffriction', 'solimp',
  'geom1', 'geom2', 'geom',
)
"""Contact fields of `mjx.Data`, placed into MJX's contact slots."""

CONSTRAINT_FIELDS = (
  'efc_J', 'efc_pos', 'efc_margin', 'efc_frictionloss', 'efc_D', 'efc_aref', 'efc_force',
)
"""Constraint fields of `mjx.Data`, with rows moved to MJX's offsets."""

CONTACT_KEY_PREFIX = '._impl.contact.'
"""Key path prefix of the contact leaves of `mjx.Data`."""

Mode = T.Literal['step', 'forward']

LeafRule = T.Literal[
  'copy',          # Same size in both layouts: a reshaped copy.
  'scalar',        # A non-array `MjData` attribute, such as `time`.
  'contact',       # A contact field, placed into MJX's contact slots.
  'constraint',    # A constraint field, with rows moved to MJX's offsets.
  'efc_type',      # Constraint row types, fixed by the model in MJX's layout.
  'solver_niter',  # Solver iterations of the first constraint island (MJX has no islands).
  'sparse',        # A sparse native matrix, densified.
  'symmetric',     # A symmetric native matrix stored as its lower triangle, densified.
  'cholesky',      # The Cholesky factor of the dense mass matrix.
  'empty',         # A field MJX leaves empty.
]
"""How one `mjx.Data` leaf is filled from `MjData`."""


# ==================================================================================================
# Physics

_PHYSICS_CACHE: dict[int, tuple[mujoco.MjModel, 'MujocoPhysics']] = {}


def get_physics(mj_model: mujoco.MjModel) -> 'MujocoPhysics':
  """Returns the `MujocoPhysics` of a compiled model.

  There is one per model object, so that its `MjData` pool persists across calls and traces.
  """
  entry = _PHYSICS_CACHE.get(id(mj_model))
  if entry is None or entry[0] is not mj_model:  # New, or a new model reusing a freed model's id.
    entry = _PHYSICS_CACHE[id(mj_model)] = (mj_model, MujocoPhysics(mj_model))
  return entry[1]


@dataclasses.dataclass
class LeafPlan:
  """How each leaf of an `mjx.Data` pytree is filled, decided once per pytree structure.

  Attributes:
    copied: `(leaf index, field name, leaf shape)` of the 'copy' leaves, which are copied for the
      whole batch at once.
    converted: `(leaf index, field name, rule)` of the other leaves, which are filled per element.
    copied_views: For each copied leaf, that field of every pooled `MjData`, reshaped to the leaf
      shape. `MjData` buffers never reallocate, so the views stay valid until the pool grows.
    copied_views_pool_size: Size of the pool when `copied_views` was built.
  """
  copied: list[tuple[int, str, tuple[int, ...]]]
  converted: list[tuple[int, str, LeafRule]]
  copied_views: list[list[np.ndarray]] = dataclasses.field(default_factory=list)
  copied_views_pool_size: int = 0


class MujocoPhysics:
  """Native C MuJoCo physics for MJX data in the 'jax' layout, used by `MjxEnv(backend='mujoco')`.

  `step(data)` and `forward(data)` replace `mjx.step(mjx_model, data)` and
  `mjx.forward(mjx_model, data)` for `mjx.Data` made with `impl='jax'`. Only the physics leaves JAX:
  tasks, sensors, controllers and rewards stay in JAX, so an environment compiles in seconds on CPU,
  where MJX's own physics takes hours to compile.

  The physics runs on the host through `jax.pure_callback`. The host is JAX's term for the Python
  process that launches compiled JAX code, as opposed to the device (CPU or GPU) that runs it: the
  callback pauses the compiled code, passes arrays to a Python function (here, C MuJoCo through its
  Python bindings) and resumes with the results.

  Each call, for every element of the (possibly nested) vmap batch:
  1. copies MuJoCo's integration state (`STATE_FIELDS`) into a persistent `mujoco.MjData`;
  2. runs `mj_step` or `mj_forward`;
  3. returns every leaf of the `mjx.Data` pytree in MJX's layout. Fields of the same size in both
     layouts are copied and reshaped (for example `xmat` from 9 to 3x3), sparse matrices are
     densified, contacts are placed into MJX's per-condim contact slots, and constraint rows are
     moved to MJX's offsets. This is the conversion of `mjx.put_data`, without the device transfer,
     and the result equals it leaf for leaf (see the tests). A leaf with no conversion rule raises,
     so nothing is silently dropped.

  Every batch element uses the same `mj_model`, so per-world model fields are not supported.

  Parity with the 'warp' backend: gaits are statistically equivalent (speeds, stride frequencies,
  duty factors, joint ranges). Single trajectories diverge within ~0.1 s of motion, as identical
  warp copies do. One systematic difference: warp and MJX replace the capsule-plane contact tangent
  by a world axis when the capsule is tilted > 60 deg from the plane (C MuJoCo keeps the projected
  capsule axis), which rotates the pyramidal friction cone of steep, sliding finger and toe contacts
  (google-deepmind/mujoco issue #2774, PRs #3678 and #3863).
  """

  def __init__(self, mj_model: mujoco.MjModel):
    """Initializes the physics of a compiled model."""
    self.mj_model = mj_model
    self.mj_datas: list[mujoco.MjData] = []  # One per batch element, grown on demand.
    self.leaf_plans: dict[tuple[str, ...], LeafPlan] = {}  # By the key paths of `mjx.Data`.
    self.warned_dropped_contacts = False

    # MJX's layout of contacts and constraint rows, as in `mjx.put_data`. MJX has a fixed number of
    # contact slots of each condim, and each slot owns a fixed block of constraint rows.
    self.slot_condim = mjx_collision_driver.make_condim(mj_model, impl=mjx_types.Impl.JAX)
    self.efc_type = mjx_constraint.make_efc_type(mj_model, self.slot_condim)
    self.slot_efc_address = mjx_constraint.make_efc_address(
      mj_model,
      self.slot_condim,
      self.efc_type,
    )
    self.ne, self.nf, self.nl, self.nc = mjx_constraint.counts(self.efc_type)
    self.nefc = self.ne + self.nf + self.nl + self.nc

    # Constraint rows per contact slot: a pyramidal friction cone has 2 rows per tangent direction.
    pyramidal = mj_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    self.slot_efc_rows = np.where(
      (self.slot_condim > 1) & pyramidal,
      2 * (self.slot_condim - 1),
      self.slot_condim,
    ).astype(int)

    # Slots of each condim, in order.
    self.condim_slots = {
      int(condim): np.flatnonzero(self.slot_condim == condim)
      for condim in np.unique(self.slot_condim)
    }

  def step(self, data: mjx.Data) -> mjx.Data:
    """Returns `data` after one `mj_step`, as `mjx.step` would."""
    return self._callback('step', data)

  def forward(self, data: mjx.Data) -> mjx.Data:
    """Returns `data` with its derived fields recomputed by `mj_forward`, as `mjx.forward` would."""
    return self._callback('forward', data)

  # ------------------------------------------------------------------------------------------------
  # JAX side

  def _callback(self, mode: Mode, data: mjx.Data) -> mjx.Data:
    """Runs `mode` on the host for `data`, which may be batched by vmap."""
    if data.impl != mjx_types.Impl.JAX:
      raise ValueError(f"MuJoCo physics needs `mjx.Data` made with impl='jax', got {data.impl}.")

    # The host returns every leaf of `mjx.Data`, identified by its key path.
    key_leaves, treedef = jax.tree_util.tree_flatten_with_path(data)
    keys = tuple(jax.tree_util.keystr(key) for key, _ in key_leaves)
    shapes = tuple(jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for _, x in key_leaves)

    # Only the integration state goes to the host. With `expand_dims`, a vmapped call stays one
    # callback that receives the whole batch.
    state = [jnp.asarray(getattr(data, name)) for name in STATE_FIELDS]
    leaves = jax.pure_callback(
      functools.partial(self._run_on_host, mode, keys, shapes),
      shapes,
      *state,
      vmap_method='expand_dims',
    )
    return jax.tree_util.tree_unflatten(treedef, leaves)

  # ------------------------------------------------------------------------------------------------
  # Host side

  def _run_on_host(
    self,
    mode: Mode,
    keys: tuple[str, ...],
    shapes: tuple[jax.ShapeDtypeStruct, ...],
    *state: np.ndarray,
  ) -> list[np.ndarray]:
    """Runs `mode` on every batch element and returns the leaves of the resulting `mjx.Data`.

    Args:
      mode: Whether to run `mj_step` or `mj_forward`.
      keys: Key path of each leaf of `mjx.Data`.
      shapes: Unbatched shape and dtype of each leaf.
      *state: Values of `STATE_FIELDS`, each `[*batch, ...]`. A batch axis that an input was not
        vmapped over has size 1.

    Returns:
      Each leaf, `[*batch, ...]`.
    """
    # Flatten the batch axes of the state into one axis of `size` elements. `qpos` is
    # `[*batch, nq]`, so it gives the number of batch axes.
    nbatch_axes = np.ndim(state[1]) - 1
    batch_shape = np.broadcast_shapes(*(np.shape(x)[:nbatch_axes] for x in state))
    size = int(np.prod(batch_shape))
    state = tuple(
      np.broadcast_to(x, batch_shape + np.shape(x)[nbatch_axes:]).reshape(
        (size,) + np.shape(x)[nbatch_axes:]
      ) for x in state
    )

    # Grow the `MjData` pool to one per element.
    while len(self.mj_datas) < size:
      self.mj_datas.append(mujoco.MjData(self.mj_model))

    # Run the physics: load each element's state into its `MjData` and step it.
    run = mujoco.mj_step if mode == 'step' else mujoco.mj_forward
    for i in range(size):
      mj_data = self.mj_datas[i]
      for name, value in zip(STATE_FIELDS, state):
        if name == 'time':
          mj_data.time = float(value[i])  # A float attribute, not an array.
        else:
          getattr(mj_data, name)[...] = value[i]
      run(self.mj_model, mj_data)

    # Plan how each leaf is filled (once per `mjx.Data` structure).
    plan = self.leaf_plans.get(keys)
    if plan is None:
      plan = self.leaf_plans[keys] = self._plan_leaves(keys, shapes)
    leaves = [np.empty((size,) + shape.shape, shape.dtype) for shape in shapes]

    # Copy the leaves of the same size in both layouts, one batched copy per leaf.
    if plan.copied_views_pool_size != len(self.mj_datas):
      plan.copied_views = [
        [getattr(mj_data, name).reshape(shape) for mj_data in self.mj_datas]
        for _, name, shape in plan.copied
      ]
      plan.copied_views_pool_size = len(self.mj_datas)
    for (index, _, _), views in zip(plan.copied, plan.copied_views):
      np.stack(views[:size], out=leaves[index])

    # Convert the other leaves, element by element.
    for i in range(size):
      mj_data = self.mj_datas[i]
      contacts, slot_contact = self._place_contacts(mj_data)
      constraints = self._place_constraints(mj_data, slot_contact)
      for index, name, rule in plan.converted:
        leaves[index][i] = self._convert_leaf(rule, name, mj_data, contacts, constraints)

    # Restore the batch axes.
    return [leaf.reshape(batch_shape + shape.shape) for leaf, shape in zip(leaves, shapes)]

  # ------------------------------------------------------------------------------------------------
  # Leaf conversion

  def _plan_leaves(
    self,
    keys: tuple[str, ...],
    shapes: tuple[jax.ShapeDtypeStruct, ...],
  ) -> LeafPlan:
    """Assigns each leaf of `mjx.Data` its `LeafRule`.

    Raises:
      ValueError: If a leaf has no rule.
    """
    mj_data = self.mj_datas[0]
    plan = LeafPlan(copied=[], converted=[])
    for index, (key, shape) in enumerate(zip(keys, shapes)):
      name = key.split('.')[-1]
      native = getattr(mj_data, name, None)
      size = int(np.prod(shape.shape))
      native_size = None if native is None else int(np.size(native))

      # Fields whose layout differs whatever their size.
      if key.startswith(CONTACT_KEY_PREFIX):
        rule = 'contact'
      elif name in CONSTRAINT_FIELDS:
        rule = 'constraint'
      elif name in ('efc_type', 'solver_niter'):
        rule = name

      # Fields of the same size in both layouts.
      elif isinstance(native, np.ndarray) and native_size == size:
        plan.copied.append((index, name, shape.shape))
        continue
      elif native is not None and native_size == size:
        rule = 'scalar'

      # Matrices that MJX stores dense and native MuJoCo may not.
      elif name in ('actuator_moment', 'ten_J'):
        rule = 'sparse'
      elif name == 'M':
        rule = 'symmetric'
      elif name == 'qLD':
        rule = 'cholesky'
      elif name == 'qLDiagInv' and size == 0:
        rule = 'empty'

      else:
        raise ValueError(
          f'{key}: native size {native_size}, MJX shape {shape.shape}; no conversion rule.'
        )
      plan.converted.append((index, name, rule))
    return plan

  def _convert_leaf(
    self,
    rule: LeafRule,
    name: str,
    mj_data: mujoco.MjData,
    contacts: dict[str, np.ndarray],
    constraints: dict[str, np.ndarray],
  ) -> np.ndarray:
    """Returns the leaf `name` of one element, filled by `rule`.

    Args:
      rule: How the leaf is filled.
      name: Field name of the leaf.
      mj_data: The element's data, after the physics ran.
      contacts: The element's contact fields in MJX's layout (`_place_contacts`).
      constraints: The element's constraint fields in MJX's layout (`_place_constraints`).
    """
    m = self.mj_model
    match rule:
      case 'scalar':
        return np.asarray(getattr(mj_data, name))
      case 'contact':
        return contacts[name]
      case 'constraint':
        return constraints[name]
      case 'efc_type':
        return self.efc_type
      case 'solver_niter':
        return mj_data.solver_niter[0]
      case 'sparse' if name == 'actuator_moment':  # Sparsity pattern in `MjData`.
        return _densify(
          mujoco.mju_sparse2dense,
          (m.nu, m.nv),
          mj_data.actuator_moment,
          mj_data.moment_rownnz,
          mj_data.moment_rowadr,
          mj_data.moment_colind,
        )
      case 'sparse' if name == 'ten_J':  # Sparsity pattern in `MjModel`.
        if not m.ntendon:
          return np.zeros((0, m.nv))
        return _densify(
          mujoco.mju_sparse2dense,
          (m.ntendon, m.nv),
          mj_data.ten_J,
          m.ten_J_rownnz,
          m.ten_J_rowadr,
          m.ten_J_colind,
        )
      case 'symmetric':
        return self._dense_mass_matrix(mj_data)
      case 'cholesky':
        # As `mjx.put_data` stores it for dense models; zero if not positive definite.
        try:
          return scipy.linalg.cho_factor(self._dense_mass_matrix(mj_data))[0]
        except scipy.linalg.LinAlgError:
          return np.zeros((m.nv, m.nv))
      case 'empty':
        return np.zeros(())
      case _:
        raise ValueError(f'{name}: unknown rule {rule!r}.')

  def _dense_mass_matrix(self, mj_data: mujoco.MjData) -> np.ndarray:
    """Returns the mass matrix `M` of `mj_data` as a dense `[nv, nv]` array."""
    m = self.mj_model
    return _densify(
      mujoco.mju_sym2dense,
      (m.nv, m.nv),
      mj_data.M,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
    )

  # ------------------------------------------------------------------------------------------------
  # Contacts and constraints

  def _place_contacts(self, mj_data: mujoco.MjData) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Places native contacts into MJX's contact slots, as `mjx._src.io._put_contact`.

    The native contacts of each condim fill that condim's slots in order. Contacts beyond the slots
    are left out, with a warning (`mjx.put_data` raises instead); the C physics still used them.

    Returns:
      The contact fields, `[nslot, ...]`: empty slots have distance 1e10 and zero elsewhere. And the
      native contact index of each slot, `[nslot]` (-1 for empty slots).
    """
    # Match native contacts to slots, condim by condim.
    contact_condim = np.asarray(mj_data.contact.dim)
    slot_contact = -np.ones(self.slot_condim.size, dtype=int)
    ndropped = int(np.isin(contact_condim, list(self.condim_slots), invert=True).sum())
    for condim, slots in self.condim_slots.items():
      contacts = np.flatnonzero(contact_condim == condim)
      nplaced = min(len(contacts), len(slots))
      slot_contact[slots[:nplaced]] = contacts[:nplaced]
      ndropped += len(contacts) - nplaced
    if ndropped and not self.warned_dropped_contacts:
      warnings.warn(
        f'{ndropped} contacts did not fit in the MJX contact slots, so the returned contact fields '
        f'leave them out (the physics used them). Warning once per model.'
      )
      self.warned_dropped_contacts = True

    # Gather each field into the slots.
    filled = slot_contact >= 0
    source = np.where(filled, slot_contact, 0)
    placed = {}
    for name in CONTACT_FIELDS:
      native = np.asarray(getattr(mj_data.contact, name))
      if name == 'frame':
        native = native.reshape((-1, 3, 3))
      if native.shape[0] == 0:
        native = np.zeros((1,) + native.shape[1:], native.dtype)  # Something to gather from.
      value = native[source]
      value[~filled] = 1e10 if name == 'dist' else 0
      placed[name] = value
    return placed, slot_contact

  def _place_constraints(
    self,
    mj_data: mujoco.MjData,
    slot_contact: np.ndarray,
  ) -> dict[str, np.ndarray]:
    """Moves native constraint rows to MJX's offsets, as `mjx._src.io._put_data_jax`.

    Args:
      mj_data: The data, after the physics ran.
      slot_contact: Native contact index of each MJX contact slot, `[nslot]` (-1 for empty slots).

    Returns:
      The constraint fields, `[nefc, ...]`, zero in rows with no native row.
    """
    m = self.mj_model

    # Native fields, with the Jacobian dense `[nefc, nv]`.
    native = {name: np.asarray(getattr(mj_data, name)) for name in CONSTRAINT_FIELDS}
    if mujoco.mj_isSparse(m):
      efc_J = np.zeros((mj_data.nefc, m.nv))
      if mj_data.nefc:
        mujoco.mju_sparse2dense(
          efc_J,
          native['efc_J'],
          mj_data.efc_J_rownnz,
          mj_data.efc_J_rowadr,
          mj_data.efc_J_colind,
        )
      native['efc_J'] = efc_J
    else:
      native['efc_J'] = native['efc_J'].reshape((-1, m.nv)) if m.nv else np.zeros((0, 0))

    # The native source row of each MJX row (-1: none). Equality, friction and limit rows keep
    # their order at the start of each layout.
    source = -np.ones(self.nefc, dtype=np.int64)
    for nrows, mjx_start, native_start in (
      (mj_data.ne, 0, 0),
      (mj_data.nf, self.ne, mj_data.ne),
      (mj_data.nl, self.ne + self.nf, mj_data.ne + mj_data.nf),
    ):
      source[mjx_start:mjx_start + nrows] = np.arange(native_start, native_start + nrows)

    # Contact rows move with their contact: each filled slot's block of rows takes the rows of its
    # native contact, row by row.
    slots = np.flatnonzero(slot_contact >= 0)
    native_address = np.asarray(mj_data.contact.efc_address)[slot_contact[slots]]
    active = native_address >= 0  # Contacts beyond their margin have no rows.
    slots, native_address = slots[active], native_address[active]
    nrows = self.slot_efc_rows[slots]
    row_in_block = np.arange(nrows.sum()) - np.repeat(np.cumsum(nrows) - nrows, nrows)
    mjx_rows = np.repeat(self.slot_efc_address[slots], nrows) + row_in_block
    source[mjx_rows] = np.repeat(native_address, nrows) + row_in_block

    # Gather each field into its rows.
    filled = source >= 0
    take = np.where(filled, source, 0)
    placed = {}
    for name in CONSTRAINT_FIELDS:
      value = native[name]
      if value.shape[0] == 0:
        value = np.zeros((1,) + value.shape[1:])  # Something to gather from.
      value = value[take]
      value[~filled] = 0.
      placed[name] = value
    return placed


# ==================================================================================================
# Helpers


def _densify(
  convert: T.Callable[..., None],
  shape: tuple[int, ...],
  *args: np.ndarray,
) -> np.ndarray:
  """Runs a `mju_*2dense` conversion into a new float64 array (MJX leaves may be float32)."""
  out = np.zeros(shape)
  convert(out, *args)
  return out
