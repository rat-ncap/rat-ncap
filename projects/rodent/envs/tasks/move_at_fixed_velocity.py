import collections
import typing as T

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

import composer as cx
import composer.utils.rewards as cxr

from .. import arenas, bodies

__all__ = [
  'MoveAtFixedVelocity',
  'MoveAtFixedVelocity_TrackSmoothState',
  'MoveAtFixedVelocity_TrackSmooth',
]

Body = bodies.Rat


class MoveAtFixedVelocity[LocalState: cx.BaseLocalState | None](cx.MjxTask[LocalState]):
  """Base task for moving at a fixed velocity while keeping balance: the body on the arena, with
  episodes that terminate when the body falls.

  An episode terminates when the head's roll or pitch exceeds its limit, or when any of the body's
  termination contact geoms (`get_termination_contact_geoms`) touches the arena's floor. Subclasses
  define the rewards, subtasks and observations.
  """
  # Configuration.
  _termination_roll: float = eqx.field(static=True)  # (deg)
  _termination_pitch: float = eqx.field(static=True)  # (deg)

  # Model elements.
  _contact_sensors: list[mj.MjsSensor] = eqx.field(static=True)

  def __init__(
    self,
    body: cx.MjxEntity,
    arena: cx.MjxEntity | None = None,
    termination_roll: float = 45.,  # (deg)
    termination_pitch: float = 45.,  # (deg)
    mj_model_nconmax: int | None = 100,
    mj_model_njmax: int | None = 200,
    **kwargs,
  ):
    """Initializes the task.

    Args:
      body: Body entity. Attached without a name prefix, as its skin's bones would not be prefixed.
      arena: Arena entity, attached with the prefix 'arena/'. It must have a geom named 'floor'.
        Defaults to a flat `arenas.Floor`.
      termination_roll: Head roll beyond which the episode terminates (deg, in (0, 180]).
      termination_pitch: Head pitch beyond which the episode terminates (deg, in (0, 180]).
      mj_model_nconmax: Maximum contacts per world, or `None` for MJX's default.
      mj_model_njmax: Maximum constraint rows per world, or `None` for MJX's default.
      **kwargs: Arguments of `cx.MjxTask` (e.g. `env_timestep`, `sim_timestep`, `local_state`).
    """
    # Head orientation: roll = right side down (+wx), pitch = front down (+wy).
    assert 0 < termination_roll <= 180
    assert 0 < termination_pitch <= 180
    self._termination_roll = termination_roll
    self._termination_pitch = termination_pitch

    arena = arena or arenas.Floor()

    mj_spec = mj.MjSpec()
    # Fewer solver iterations than MuJoCo's defaults (100 and 50), for faster simulation.
    mj_spec.option.iterations = 10
    mj_spec.option.ls_iterations = 20
    mj_spec.visual = arena.mj_spec.visual
    mj_spec.attach(arena.mj_spec, 'arena/', frame=mj_spec.worldbody.add_frame())
    # The body is attached without a prefix, as `attach()` does not prefix the body names of a
    # skin's bones.
    # Without a prefix, `attach()` leaves the body's root default class unnamed (it would be
    # `<prefix>main`), which `to_xml()` writes as a nested `<default>` that MuJoCo cannot parse
    # back (MuJoCo issue #2707). Naming it, as the arena's is named `arena/main`, keeps the spec's
    # XML loadable.
    body.mj_spec.default.name = 'body/main'
    mj_spec.attach(body.mj_spec, '', frame=mj_spec.worldbody.add_frame())

    # Add contact termination sensors.
    self._contact_sensors = []
    for geom in body.get_termination_contact_geoms():
      self._contact_sensors.append(
        mj_spec.add_sensor(
          name=f'contact_floor_{geom}',
          type=mj.mjtSensor.mjSENS_CONTACT,
          objtype=mj.mjtObj.mjOBJ_GEOM,
          objname=geom,
          reftype=mj.mjtObj.mjOBJ_GEOM,
          refname='arena/floor',
          intprm=[1, 1, 1],  # [data='found'(00000001), reduce='mindist'(enum 1), num=1]
        )
      )

    super().__init__(
      mj_spec=mj_spec,
      children={
        'arena': arena,
        'body': body,
      },
      mj_model_nconmax=mj_model_nconmax,
      mj_model_njmax=mj_model_njmax,
      **kwargs,
    )

  @T.override
  def on_reset(
    self,
    state: cx.MjxEnvState,
    subtask: cx.Subtask,
  ) -> cx.MjxEnvState:
    """Resets the episode with the body in its initial pose."""
    state = super().on_reset(state, subtask)
    body = T.cast(Body, self.children['body'])
    state = body.init_pose(state)
    return state

  @T.override
  def get_terminated(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns whether the head is rolled or pitched beyond its limit, or a termination contact
    geom touches the floor."""
    body = T.cast(Body, self.children['body'])
    rpy = body.get_sensor_rpy(state)
    terminated = jnp.logical_or(
      jnp.abs(rpy[0]) > np.deg2rad(self._termination_roll),
      jnp.abs(rpy[1]) > np.deg2rad(self._termination_pitch),
    )
    if len(self._contact_sensors) > 0:
      contacts = state.mjx_data.bind(state.mjx_model, self._contact_sensors).sensordata  # [sensors]
      terminated |= contacts.any()
    return terminated


class MoveAtFixedVelocity_TrackSmoothState(cx.BaseLocalState):
  """Local state of `MoveAtFixedVelocity_TrackSmooth`: the target velocity and the subtask reward
  weights."""
  target_vx: jax.Array  # [], float (m/s)

  reward_passive: jax.Array  # [], float
  reward_vx_sparse: jax.Array  # [], float
  reward_vx_dense: jax.Array  # [], float
  reward_pose: jax.Array  # [], float


class MoveAtFixedVelocity_TrackSmooth(MoveAtFixedVelocity[MoveAtFixedVelocity_TrackSmoothState]):
  """Reward function based on "Learning to Walk in Minutes Using Massively Parallel Deep RL" [Rudin
  et al. 2022] (https://arxiv.org/pdf/2109.11978) and "Barkour: Benchmarking Animal-Level Agility
  with Quadruped Robots" [Caluwaerts et al. 2023] (https://arxiv.org/pdf/2305.14654).

  Each subtask sets a target forward velocity `target_vx` (global x, m/s) and the weights of four
  reward terms:
  - `reward_passive`: 1 every step.
  - `reward_vx_sparse`: 1 at `target_vx`, falling to 0 at `margin_vx` from it.
  - `reward_vx_dense`: 1 at `target_vx`, falling linearly to 0 at `max(target_vx, margin_vx)` from
    it, so at 0 m/s for a target of at least `margin_vx`.
  - `reward_pose`: while standing (`target_vx` is 0, including during `stand_time`), 1 at the
    initial pose of the body's pose joints, decaying with their largest deviation. While moving,
    it is 1.

  Further joint-space and world-space regularization terms (`punish_*`) have constant weights,
  0 by default.
  """
  # Task hyperparameters.
  _stand_time: float = eqx.field(static=True)
  _min_vx: float = eqx.field(static=True)
  _max_vx: float = eqx.field(static=True)


  # Reward shaping hyperparameters.
  _normalize_reward: bool = eqx.field(static=True)
  _optimal_vx: float = eqx.field(static=True)
  _margin_vx: float = eqx.field(static=True)
  _margin_pose: float = eqx.field(static=True)
  _margin_qwork: float = eqx.field(static=True)
  _margin_slip_vx: float = eqx.field(static=True)
  _reward_slip_vx_airtime: float = eqx.field(static=True)

  # Reward weight hyperparameters.
  _punish_vy: float = eqx.field(static=True)
  _punish_vz: float = eqx.field(static=True)
  _punish_wx: float = eqx.field(static=True)
  _punish_wy: float = eqx.field(static=True)
  _punish_wz: float = eqx.field(static=True)
  _punish_ax: float = eqx.field(static=True)
  _punish_roll: float = eqx.field(static=True)
  _punish_pitch: float = eqx.field(static=True)
  _punish_qvel: float = eqx.field(static=True)
  _punish_qtrq: float = eqx.field(static=True)
  _punish_qwork: float = eqx.field(static=True)
  _punish_slip_vx: float = eqx.field(static=True)

  def __init__(
    self,
    body: bodies.Rat,
    *,
    # Task hyperparameters.
    stand_time: float = 0.,
    min_vx: float = -5.0,
    max_vx: float = 5.0,


    # Reward shaping hyperparameters.
    normalize_reward: bool = False,
    optimal_vx: float = 0.0,
    margin_vx: float = 0.1,
    margin_pose: float = 10.,
    margin_qwork: float = 0.3,
    margin_slip_vx: float = 0.1,
    reward_slip_vx_airtime: float = 0.0,

    # Reward weight hyperparameters.
    punish_vy: float = 0.,
    punish_vz: float = 0.,
    punish_wx: float = 0.,
    punish_wy: float = 0.,
    punish_wz: float = 0.,
    punish_ax: float = 0.,
    punish_roll: float = 0.,
    punish_pitch: float = 0.,
    punish_qvel: float = 0.,
    punish_qtrq: float = 0.,
    punish_qwork: float = 0.,
    punish_slip_vx: float = 0.,
    **kwargs,
  ):
    """Args:
      body: Robot entity passed to `MoveAtFixedVelocity`.
      stand_time: Time since reset during which the target velocity is 0 (s).
      min_vx: Minimum target velocity along the global x-axis (m/s).
      max_vx: Maximum target velocity along the global x-axis (m/s).
      normalize_reward: Whether to normalize the positive reward weights to sum to 1 (see
        `get_reward`).
      optimal_vx: Width of the optimal range of `reward_vx_sparse`, from `target_vx` to
        `target_vx + optimal_vx` (m/s).
      margin_vx: Margin for zero `reward_vx_sparse`, and least margin for zero `reward_vx_dense`
        (m/s).
      margin_pose: `reward_pose` margin (deg).
      margin_qwork: `punish_qwork` margin (N*m/s).
      margin_slip_vx: `punish_slip_vx` margin (m/s).
      reward_slip_vx_airtime: Value used for `punish_slip_vx` when the foot is in the air.
        If the airtime reward were 0, the agent would be incentivized to never lift its feet.
      punish_vy: Penalty for the root's linear velocity, in the global frame (`vy` -> 0).
        `vy` = +left/-right.
      punish_vz: Penalty for the root's linear velocity (`vz` -> 0).
        `vz` = +up/-down.
      punish_wx: Penalty for the root's angular velocity, in the global frame (`wx` -> 0).
        `wx` = +right-goes-down. (roll velocity)
      punish_wy: Penalty for the root's angular velocity (`wy` -> 0).
        `wy` = +front-goes-down. (pitch velocity)
      punish_wz: Penalty for the root's angular velocity (`wz` -> 0).
        `wz` = +front-goes-left. (yaw velocity)
      punish_ax: Penalty for the root's linear acceleration (`ax` -> 0).
      punish_roll: Penalty for the head's roll (`roll` -> 0).
      punish_pitch: Penalty for the head's pitch (`pitch` -> 0).
      punish_qvel: Penalty for joint velocity (`qvel` -> 0).
      punish_qtrq: Penalty for joint torques (`qtrq` -> 0).
      punish_qwork: Penalty for joint work (`qtrq * qvel` -> 0).
      punish_slip_vx: Penalty for slip velocity in the x-direction (`slip_vx` -> 0).
      **kwargs: Arguments passed to `MoveAtFixedVelocity`.
    """
    kwargs.setdefault('local_state', MoveAtFixedVelocity_TrackSmoothState(
        target_vx=jnp.zeros((), dtype=jnp.float32),
        reward_passive=jnp.zeros((), dtype=jnp.float32),
        reward_vx_sparse=jnp.zeros((), dtype=jnp.float32),
        reward_vx_dense=jnp.zeros((), dtype=jnp.float32),
        reward_pose=jnp.zeros((), dtype=jnp.float32),
      )
    )
    super().__init__(body, **kwargs)

    self._stand_time = stand_time
    self._min_vx = min_vx
    self._max_vx = max_vx


    self._normalize_reward = normalize_reward
    self._optimal_vx = optimal_vx
    self._margin_vx = margin_vx
    self._margin_pose = margin_pose
    self._margin_qwork = margin_qwork
    self._margin_slip_vx = margin_slip_vx
    self._reward_slip_vx_airtime = reward_slip_vx_airtime

    self._punish_vy = punish_vy
    self._punish_vz = punish_vz
    self._punish_wx = punish_wx
    self._punish_wy = punish_wy
    self._punish_wz = punish_wz
    self._punish_ax = punish_ax
    self._punish_roll = punish_roll
    self._punish_pitch = punish_pitch
    self._punish_qvel = punish_qvel
    self._punish_qtrq = punish_qtrq
    self._punish_qwork = punish_qwork
    self._punish_slip_vx = punish_slip_vx

  @T.override
  def on_reset(
    self,
    state: cx.MjxEnvState,
    subtask: cx.Subtask,
  ) -> cx.MjxEnvState:
    """Resets the episode with the subtask's target velocity and reward weights."""
    state = super().on_reset(state, subtask)

    local = state.local.get(self.index)
    local = local.replace(target_vx=subtask['target_vx'])
    local = local.replace(
      reward_passive=subtask['reward_passive'],
      reward_vx_sparse=subtask['reward_vx_sparse'],
      reward_vx_dense=subtask['reward_vx_dense'],
      reward_pose=subtask['reward_pose'],
    )

    state = state.replace(local=state.local.set(self.index, local))

    return state

  @T.override
  def get_rewards(self, state: cx.MjxEnvState, terminated: jax.Array) -> cx.MjxTaskRewards:
    """Returns the velocity tracking, pose, passive and regularization reward terms as
    `{name: (weight, value)}`."""
    body = T.cast(bodies.Rat, self.children['body'])
    local = state.local.get(self.index)

    # Gather local state.
    reward_passive = local.reward_passive
    reward_vx_sparse = local.reward_vx_sparse
    reward_vx_dense = local.reward_vx_dense
    reward_pose = local.reward_pose

    # Gather sensor data.
    vx, vy, vz = body.get_root_global_linvel(state)
    wx, wy, wz = body.get_root_global_angvel(state)
    ax, _, _ = body.get_root_global_linacc(state)
    roll, pitch, _ = body.get_sensor_rpy(state)
    qvel = body.get_proprio_vel(state)
    qtrq = body.get_proprio_trq(state)

    # Gather target data.
    target_vx = self.get_target_vx(state)
    optimal_vx = self._optimal_vx
    standing = target_vx == 0

    # Compute pose error.
    pose_standing = body.get_pose_init()
    pose_current = body.get_pose(state)
    pose_target = pose_standing * standing + pose_current * ~standing
    pose_err = pose_current - pose_target

    # Compute slipx error.
    digit_linvel = body.get_foot_global_linvel(state)  # [4, 3]
    tactile_foot = body.get_tactile_foot(state)  # [8] (digit, paw, digit, paw, ...)

    digit_vx = digit_linvel[:, 0]  # [4]
    tactile_digit = tactile_foot.reshape((4, 2))[:, 0]  # [4]
    slip_vx = digit_vx
    slip_vx_mask = tactile_digit > 0

    # Construct reward terms.
    rewards = {
      'reward_passive': (+reward_passive, jnp.ones(())),
      'reward_vx_sparse': (
        +reward_vx_sparse,
        cxr.reward_curve(
          vx,
          optimal=(target_vx, target_vx + optimal_vx),
          margin=self._margin_vx,
          curve='concave'
        ),
      ),
      'reward_vx_dense': (
        +reward_vx_dense,
        cxr.reward_curve(
          vx,
          optimal=target_vx,
          margin=jnp.maximum(target_vx, self._margin_vx),
          curve='linear',
        ),
      ),
      'reward_pose': (
        +reward_pose,
        cxr.reward_curve(
          jnp.abs(pose_err).max(),  # The greatest angle deviation should not exceed the margin.
          optimal=0.0,
          margin=jnp.deg2rad(self._margin_pose),
          curve='long_tail',
        )
      ),
      'punish_vy': (-jnp.array(self._punish_vy), vy**2),
      'punish_vz': (-jnp.array(self._punish_vz), vz**2),
      'punish_wx': (-jnp.array(self._punish_wx), wx**2),
      'punish_wy': (-jnp.array(self._punish_wy), wy**2),
      'punish_wz': (-jnp.array(self._punish_wz), wz**2),
      'punish_ax': (-jnp.array(self._punish_ax), ax**2),
      'punish_roll': (-jnp.array(self._punish_roll), roll**2),
      'punish_pitch': (-jnp.array(self._punish_pitch), pitch**2),
      'punish_qvel': (-jnp.array(self._punish_qvel), (qvel * qvel).mean()),
      'punish_qtrq': (-jnp.array(self._punish_qtrq), (qtrq * qtrq).mean()),
      'punish_qwork': (
        +jnp.array(self._punish_qwork),
        cxr.reward_curve(
          (qtrq * qvel).mean(),
          optimal=0,
          margin=self._margin_qwork,
          curve='long_tail',
        )
      ),
      'punish_slip_vx': (
        +jnp.array(self._punish_slip_vx),
        (cxr.reward_curve(
          slip_vx,
          optimal=0.0,
          margin=self._margin_slip_vx,
          curve='long_tail',
        ) * slip_vx_mask + self._reward_slip_vx_airtime * ~slip_vx_mask).mean()
      ),
    }
    return rewards

  @T.override
  def get_reward(self, rewards: cx.MjxTaskRewards) -> jax.Array:
    """Returns the weighted sum of the reward terms, with the positive weights normalized to sum to
    1 if `normalize_reward`."""
    normalize = sum((w.clip(0, 1) for w, _ in rewards.values())) if self._normalize_reward else 1.0
    return sum((w * v for w, v in rewards.values()), jnp.zeros(())) / normalize

  @T.override
  def subtask_space(self, state: cx.MjxEnvState) -> cx.SubtaskSpace:
    """Returns the subtask space: the target velocity and the reward weights."""
    space = cx.Dict()
    space['target_vx'] = cx.Box(
      low=self._min_vx,
      high=self._max_vx,
      shape=(),
      name='target_vx:[tvx]',
      desc='Target linear velocity along x-axis in global frame (m/s)',
    )
    space['reward_passive'] = cx.Box(
      low=0.0,
      high=1.0,
      shape=(),
      name='reward_passive',
      desc='Passive reward weight (unitless)',
    )
    space['reward_vx_sparse'] = cx.Box(
      low=0.0,
      high=1.0,
      shape=(),
      name='reward_vx_sparse',
      desc='Sparse reward weight for forward velocity tracking (unitless)',
    )
    space['reward_vx_dense'] = cx.Box(
      low=0.0,
      high=1.0,
      shape=(),
      name='reward_vx_dense',
      desc='Dense reward weight for forward velocity tracking (unitless)',
    )
    space['reward_pose'] = cx.Box(
      low=0.0,
      high=1.0,
      shape=(),
      name='reward_pose',
      desc='Reward weight for pose tracking (unitless)',
    )
    return space

  @T.override
  def observation_space(self, state: cx.MjxEnvState) -> cx.ObservationSpace:
    """Adds the target velocity to the observation space."""
    del state
    space = cx.Dict()
    space['target_vx'] = cx.Box(
      low=self._min_vx,
      high=self._max_vx,
      shape=(1,),
      name='target_vx:[tvx]',
      desc='Target linear velocity along x-axis in global frame (m/s)',
    )
    return space

  @T.override
  def observation(self, state: cx.MjxEnvState) -> tuple[cx.ArrayDict, cx.MjxEnvState]:
    """Adds the target velocity to the observation, 0 during the first `stand_time` seconds."""
    observation = collections.OrderedDict()
    observation['target_vx'] = self.get_target_vx(state).reshape((1,))
    return observation, state

  def get_target_vx(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the target velocity: 0 during the first `stand_time` seconds, then the subtask's
    (m/s), []."""
    # `<`, as `<=` would make it 0 on the first step even with a `stand_time` of 0.
    return jnp.where(
      state.mjx_data.time < self._stand_time,
      0.,
      state.local.get(self.index).target_vx,
    )
