import collections
import collections.abc
import os
import pathlib
import types
import typing as T

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

import composer as cx
import composer.math as cxm
import composer.transforms as cxt

__all__ = [
  'RAT_XML', 'RatSegment', 'RAT_ACTUATORS', 'RAT_ACTUATOR_NAMES', 'RAT_VEL_MAX', 'RAT_TRQ_MAX',
  'RAT_FOOT_MAX', 'RAT_ANGVEL_MAX', 'RAT_LINACC_MAX', 'RAT_ACTUATORS_POS_MID',
  'RAT_ACTUATORS_POS_MID_01', 'RAT_ACTUATORS_SETPOINT', 'RAT_ACTUATORS_DELTA',
  'RAT_ACTUATORS_POS_MIN', 'RAT_ACTUATORS_POS_MAX', 'RAT_ACTUATORS_KP', 'RAT_ACTUATORS_KD',
  'RAT_ACTUATORS_TRQ',
  'rat_actuators_pos_mid', 'rat_actuators_pos_mid_01', 'RAT_ROOT_POS_INIT', 'RAT_ROOT_QUAT_INIT',
  'RAT_JOINTS_POS_INIT', 'POSE_JOINTS', 'JointPassive', 'RatState', 'Rat',
]
# Path to the rat's MuJoCo XML file.
RAT_XML = pathlib.Path(__file__).parent / 'rat.xml'

RatSegment = T.Literal[
  'head',
  'cervicalE',
  'cervicalBT',
  'limbFL',
  'limbFR',
  'lumbarE',
  'lumbarBT',
  'limbHL',
  'limbHR',
  'tailE',
  'tailB',
]
RAT_ACTUATORS = T.cast(T.Mapping[RatSegment, T.Tuple[str, ...]], types.MappingProxyType(dict(
  head=('mandible', 'atlas'),
  cervicalE=('cervical_extend',),
  cervicalBT=('cervical_bend', 'cervical_twist'),
  limbFL=(
    'scapula_L_supinate',
    'scapula_L_abduct',
    'scapula_L_extend',
    'shoulder_L_supinate',
    'shoulder_L_extend',
    'elbow_L',
    'wrist_L',
    'finger_L'
  ),
  limbFR=(
    'scapula_R_supinate',
    'scapula_R_abduct',
    'scapula_R_extend',
    'shoulder_R_supinate',
    'shoulder_R_extend',
    'elbow_R',
    'wrist_R',
    'finger_R'
  ),
  lumbarE=('lumbar_extend',),
  lumbarBT=('lumbar_bend', 'lumbar_twist'),
  limbHL=('hip_L_supinate', 'hip_L_abduct', 'hip_L_extend', 'knee_L', 'ankle_L', 'toe_L'),
  limbHR=('hip_R_supinate', 'hip_R_abduct', 'hip_R_extend', 'knee_R', 'ankle_R', 'toe_R'),
  tailE=('tail_extend',),
  tailB=('tail_bend',),
)))
RAT_ACTUATOR_NAMES = tuple(actuator for segment in RAT_ACTUATORS.values() for actuator in segment)

# Observation ranges for actuator velocities and stiffness/damping coefficients.
# Control ranges for joint positions can be extracted from the XML.
RAT_VEL_MAX = 20.  # (rad/s)
RAT_TRQ_MAX = 1.  # (N.m)
RAT_FOOT_MAX = 3.  # (N)
RAT_ANGVEL_MAX = 2. * jnp.pi  # (rad/s)
RAT_LINACC_MAX = 10.  # (m/s^2)

# Actuator position midpoints for a midpoint+delta action space.
RAT_ACTUATORS_POS_MID = types.MappingProxyType(dict(
  head=np.deg2rad([-36, 20]),
  cervicalE=np.array([-0.33]),
  cervicalBT=np.array([0, 0]),
  limbFL=np.deg2rad([4.5, 0, 0, -16.25, 20, -2, 61.2, -65]),
  limbFR=np.deg2rad([4.5, 0, 0, -16.25, 20, -2, 61.2, -65]),
  lumbarE=np.array([0.1]),
  lumbarBT=np.array([0, 0]),
  limbHL=np.deg2rad([-1.5, 12.75, -15, -35.2, 14, 0]),
  limbHR=np.deg2rad([-1.5, 12.75, -15, -35.2, 14, 0]),
  tailE=np.array([0]),
  tailB=np.array([0]),
))
# Actuator position midpoints normalized into [0, 1] over the actuators' position ranges widened by
# 0.2 of the range on each side (as the controllers' `pos_margin: 0.2`), from
# `RAT_ACTUATORS_POS_MID` by hand (2 decimals).
RAT_ACTUATORS_POS_MID_01 = types.MappingProxyType(dict(
  head=np.array([0.00, 0.50]),
  cervicalE=np.array([0.38]),
  cervicalBT=np.array([0.50, 0.50]),
  limbFL=np.array([0.50, 0.50, 0.50, 0.65, 0.69, 0.60, 0.67, 0.60]),
  limbFR=np.array([0.50, 0.50, 0.50, 0.65, 0.69, 0.60, 0.67, 0.60]),
  lumbarE=np.array([0.54]),
  lumbarBT=np.array([0.50, 0.50]),
  limbHL=np.array([0.54, 0.48, 0.50, 0.30, 0.27, 0.62]),
  limbHR=np.array([0.54, 0.48, 0.50, 0.30, 0.27, 0.62]),
  tailE=np.array([0.50]),
  tailB=np.array([0.50]),
))
RAT_ACTUATORS_SETPOINT = types.MappingProxyType(dict(
  head=np.deg2rad([-15, 20]),
  cervicalE=np.deg2rad([-10]),
  cervicalBT=np.deg2rad([0, 0]),
  limbFL=np.deg2rad([4.5, 0, 0, 0, 20, -2, 50, -65]),
  limbFR=np.deg2rad([4.5, 0, 0, 0, 20, -2, 50, -65]),
  lumbarE=np.deg2rad([15]),
  lumbarBT=np.deg2rad([0, 0]),
  limbHL=np.deg2rad([-5, 15, -15, 0, 36, 0]),
  limbHR=np.deg2rad([-5, 15, -15, 0, 36, 0]),
  tailE=np.deg2rad([0]),
  tailB=np.deg2rad([0]),
))
RAT_ACTUATORS_DELTA = types.MappingProxyType(dict(
  head=np.deg2rad([30, 25]),
  cervicalE=np.deg2rad([25]),
  cervicalBT=np.deg2rad([35, 35]),
  limbFL=np.deg2rad([5.5, 15, 2, 65, 95, 73, 70, 115]),
  limbFR=np.deg2rad([5.5, 15, 2, 65, 95, 73, 70, 115]),
  lumbarE=np.deg2rad([30]),
  lumbarBT=np.deg2rad([30, 30]),
  limbHL=np.deg2rad([35, 45, 75, 88, 44, 100]),
  limbHR=np.deg2rad([35, 45, 75, 88, 44, 100]),
  tailE=np.deg2rad([15]),
  tailB=np.deg2rad([15]),
))

RAT_ACTUATORS_POS_MIN = types.MappingProxyType({
  k: RAT_ACTUATORS_POS_MID[k] - RAT_ACTUATORS_DELTA[k] for k in RAT_ACTUATORS
})
RAT_ACTUATORS_POS_MAX = types.MappingProxyType({
  k: RAT_ACTUATORS_POS_MID[k] + RAT_ACTUATORS_DELTA[k] for k in RAT_ACTUATORS
})
# These ranges are symmetric about the mid pose, although several joints' ranges are not; this
# only affects the 'setpoint_delta' rescale mode.
# [scapula_{sup,abd,ext}, shoulder_{sup,ext}, elbow, wrist, finger]
# [hip_{sup,abd,ext}, knee, ankle, toe]
RAT_ACTUATORS_KP = T.cast(T.Mapping[RatSegment, T.Sequence[float]], types.MappingProxyType(dict(
  head=np.array([0.1, 0.05]),
  cervicalE=np.array([20]),
  cervicalBT=np.array([20, 20]),
  limbFL=np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1, 0.05]),
  limbFR=np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1, 0.05]),
  lumbarE=np.array([20]),
  lumbarBT=np.array([20, 20]),
  limbHL=np.array([0.4, 0.4, 0.6, 0.4, 0.4, 0.05]),
  limbHR=np.array([0.4, 0.4, 0.6, 0.4, 0.4, 0.05]),
  tailE=np.array([1]),
  tailB=np.array([2]),
)))
RAT_ACTUATORS_KD = T.cast(T.Mapping[RatSegment, T.Sequence[float]], types.MappingProxyType(dict(
  head=np.array([0.002, 0.002]),
  cervicalE=np.array([1.0]),
  cervicalBT=np.array([1.0, 1.0]),
  limbFL=np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002]),
  limbFR=np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002]),
  lumbarE=np.array([0.03]),
  lumbarBT=np.array([0.03, 0.03]),
  limbHL=np.array([0.02, 0.02, 0.02, 0.005, 0.003, 0.001]),
  limbHR=np.array([0.02, 0.02, 0.02, 0.005, 0.003, 0.001]),
  tailE=np.array([0.002]),
  tailB=np.array([0.002]),
)))
RAT_ACTUATORS_TRQ = types.MappingProxyType({
  k: RAT_TRQ_MAX * np.ones_like(RAT_ACTUATORS_KP[k]) for k in RAT_ACTUATORS
})


def rat_actuators_pos_mid(segment: RatSegment) -> np.ndarray:
  """Returns the position midpoints (rad, or m for tendons) of a segment's actuators."""
  return RAT_ACTUATORS_POS_MID[segment]


def rat_actuators_pos_mid_01(segment: RatSegment) -> np.ndarray:
  """Returns the position midpoints of a segment's actuators, normalized into [0, 1] (see
  `RAT_ACTUATORS_POS_MID_01`)."""
  return RAT_ACTUATORS_POS_MID_01[segment]


# Root position (x, y, z) in world frame (m).
RAT_ROOT_POS_INIT = np.array([0.01635, 0.00062, 0.05109])
# Root orientation (w, x, y, z) in world frame (quat).
RAT_ROOT_QUAT_INIT = np.array([0.99818, 0.00005, -0.06028, 0.0011])
# Joints initial positions corresponding to zero/standing pose after settling time.
# yapf: disable
RAT_JOINTS_POS_INIT = np.deg2rad([
   16.1,   0.0,  -0.0,  -9.4,  -0.0,   0.0,  -1.5,  15.9, -16.2, -35.1,   9.8,  14.1,  -1.5,  15.9,
  -16.2, -35.1,   9.8,  14.1,  -5.1,  -0.0,   5.6,  -0.0,  10.1,  -0.0,   1.1,  -0.0,  -2.0,  -0.0,
   -0.0,   0.0, -16.8,  -0.0,  -0.0, -16.3,  -0.0,   0.0, -15.9,  13.8, -18.4,   4.3,   1.6,  -1.8,
  -16.5,  18.7,  -4.6,  64.8, -62.2,   4.3,   1.6,  -1.8, -16.5,  18.7,  -4.6,  64.8, -62.2,
])

# yapf: enable

# Joints of the pose (`Rat.get_pose`): every joint except the root, the 7 cervical vertebrae,
# `atlas` and `mandible`.
POSE_JOINTS = [
  'vertebra_1_extend',
  'vertebra_2_bend',
  'vertebra_3_twist',
  'vertebra_4_extend',
  'vertebra_5_bend',
  'vertebra_6_twist',
  'hip_L_supinate',
  'hip_L_abduct',
  'hip_L_extend',
  'knee_L',
  'ankle_L',
  'toe_L',
  'hip_R_supinate',
  'hip_R_abduct',
  'hip_R_extend',
  'knee_R',
  'ankle_R',
  'toe_R',
  'vertebrae_C1C3_extend',
  'vertebrae_C1C3_bend',
  'vertebrae_C4C9_extend',
  'vertebrae_C4C9_bend',
  'vertebrae_C10C15_extend',
  'vertebrae_C10C15_bend',
  'vertebrae_C16C20_extend',
  'vertebrae_C16C20_bend',
  'vertebrae_C21C25_extend',
  'vertebrae_C21C25_bend',
  'vertebrae_C26C30_extend',
  'vertebrae_C26C30_bend',
  'scapula_L_supinate',
  'scapula_L_abduct',
  'scapula_L_extend',
  'shoulder_L_supinate',
  'shoulder_L_extend',
  'elbow_L',
  'wrist_L',
  'finger_L',
  'scapula_R_supinate',
  'scapula_R_abduct',
  'scapula_R_extend',
  'shoulder_R_supinate',
  'shoulder_R_extend',
  'elbow_R',
  'wrist_R',
  'finger_R'
]


class JointPassive(T.TypedDict, total=False):
  """Passive properties of a joint, overriding the XML (`Rat(joint_passive=...)`); unset keys keep
  the XML values."""
  stiffness: float
  damping: float
  armature: float


class RatState(cx.BaseLocalState):
  """Local state of `Rat`: the foot contact counters (see `Rat.update_foot_counters`)."""
  foot_contact: jax.Array  # [4], bool
  """Whether each foot was in contact with the ground in the previous env timestep."""
  foot_liftoff: jax.Array  # [4], bool
  """Whether each foot went from stance to swing (liftoff) in the current env timestep."""
  foot_liftoff_time: jax.Array  # [4], float
  """Time (s) when each foot made the most recent liftoff."""
  foot_liftoff_pos: jax.Array  # [4, 3], float
  """Position (m) in global frame of each foot at the most recent liftoff."""
  foot_touchdown: jax.Array  # [4], bool
  """Whether each foot went from swing to stance (touchdown) in the current env timestep."""
  foot_touchdown_time: jax.Array  # [4], float
  """Time (s) when each foot made the most recent touchdown."""
  foot_stance_duration: jax.Array  # [4], float
  """Duration (s) of the most recent stance phase of each foot."""
  foot_swing_duration: jax.Array  # [4], float
  """Duration (s) of the most recent swing phase of each foot."""
  foot_swing_length: jax.Array  # [4], float
  """Length (m) of the most recent swing phase of each foot."""
  foot_swing_peak: jax.Array  # [4], float
  """Peak height (m) of the most recent swing phase of each foot."""


class Rat(cx.MjxEntity[RatState]):
  """The rat body: an MJX entity of the rat model (`RAT_XML`), with its actuators grouped into
  segments (`RAT_ACTUATORS`: head, cervical, the four limbs, lumbar and tail).

  Actions are torques per segment (`actuators_trq_<segment>`), produced by the transforms (in
  order, innermost first):
  1. passive PD torques of every actuator toward its midpoint (`passive_kp`, `passive_kd`);
  2. zero torques for the actuators that are not actionable (`actionable_dofs`);
  3. the controller (e.g. a mapping of `transforms.SpringMuscleController` per segment, which takes
     `muscle_{flx,ext}_<segment>` actions), whose torques add to the passive torques;
  4. rescaling of the position observations and actions (`rescale_mode`);
  5. reducing the observations and actions to the actionable actuators.
  Observations are per segment: `proprio_{pos,vel,trq}_<segment>` (positions in rad, or tendon
  lengths in m for the cervical, lumbar and tail tendons) and `tactile_limb<foot>` (digit and paw
  contact forces).

  With a mapping controller (`controller_actkey_prefixes=()`), the controller's actions (e.g.
  `muscle_*` or `actuators_pos_*`) are not reduced to the actionable actuators: they keep every
  actuator of the segment, and step 2 zeroes the torques of the others.

  The orientation sensors (`get_sensor_quat`, `get_sensor_rpy`, and so the task's roll and pitch)
  are on the head (the `head` site of the skull), not the torso.
  """
  # Configuration.
  _actionable_dofs: T.Mapping[RatSegment, T.Sequence[str]] = eqx.field(static=True)
  _observable_proprio: bool = eqx.field(static=True)
  _observable_tactile: bool = eqx.field(static=True)
  _observable_vestibular: bool = eqx.field(static=True)
  _observable_visual: bool = eqx.field(static=True)
  _root_pos_init: jax.Array  # [3], float (m)
  _root_quat_init: jax.Array  # [4], float (quat)
  _joints_pos_init: jax.Array  # [55], float (rad)

  # Model elements.
  _actuators: list[mj.MjsActuator] = eqx.field(static=True)
  _actuateds: list[mj.MjsJoint | mj.MjsTendon] = eqx.field(static=True)

  _root_body: mj.MjsBody = eqx.field(static=True)
  _pelvis_body: mj.MjsBody = eqx.field(static=True)
  _root_joint: mj.MjsJoint = eqx.field(static=True)
  _root_local_linvel: mj.MjsSensor = eqx.field(static=True)
  _root_global_linvel: mj.MjsSensor = eqx.field(static=True)
  _root_global_angvel: mj.MjsSensor = eqx.field(static=True)
  _root_global_linacc: mj.MjsSensor = eqx.field(static=True)

  _head_quaternion: mj.MjsSensor = eqx.field(static=True)
  _head_gyroscope: mj.MjsSensor = eqx.field(static=True)
  _head_accelerometer: mj.MjsSensor = eqx.field(static=True)
  _head_global_linvel: mj.MjsSensor | None = eqx.field(static=True)

  _foot_names: tuple[str, str, str, str] = eqx.field(static=True)
  _foot_touch: list[mj.MjsSensor] = eqx.field(static=True)
  _foot_global_linvel: list[mj.MjsSensor] = eqx.field(static=True)
  _foot_global_pos: list[mj.MjsSensor] = eqx.field(static=True)

  _actuator_names: tuple[str, ...] = eqx.field(static=True)
  _termination_contact_geoms: list[str] = eqx.field(static=True)

  def __init__(
    self,
    observable_proprio: bool = True,
    observable_tactile: bool = True,
    observable_vestibular: bool = False,
    observable_visual: bool = False,
    actionable_dofs: T.Mapping[RatSegment, T.Sequence[str]] = RAT_ACTUATORS,
    contact_mode: T.Literal[
      'all',
      'trunk-limbs',
      'trunk-limbs-tail',
      'trunk-limbs-tail-head',
      'no-self',
      'foot',
    ] = 'foot',
    foot_compliance: dict[str, T.Any] | None = None,
    delete_geoms: bool = False,
    rescale_mode: T.Literal['setpoint_delta', 'limits'] = 'setpoint_delta',
    rescale_observations: bool = True,
    rescale_actions: bool = True,
    noise_observations: T.Mapping[str, T.Any] = types.MappingProxyType({}),
    noise_actions: T.Mapping[str, T.Any] = types.MappingProxyType({}),
    controller: T.Literal['position', 'position-stiffness'] | cx.Transform | T.Mapping[
      RatSegment,
      cx.Transform
    ] | None = 'position',
    controller_actkey_prefixes: T.Collection[str] = (),
    controller_obskey_prefixes: T.Collection[str] = (),
    actionable_trq: bool = False,
    transforms: T.Sequence[cx.Transform | None] = (),
    passive_kp: T.Mapping[RatSegment, T.Sequence[float]] = RAT_ACTUATORS_KP,
    passive_kd: T.Mapping[RatSegment, T.Sequence[float]] = RAT_ACTUATORS_KD,
    root_pos_init: jax.typing.ArrayLike = RAT_ROOT_POS_INIT,
    root_quat_init: jax.typing.ArrayLike = RAT_ROOT_QUAT_INIT,
    joints_pos_init: jax.typing.ArrayLike = RAT_JOINTS_POS_INIT,
    xml_path: os.PathLike | str = RAT_XML,
    head_global_linvel_sensor: bool = False,
    joint_passive: T.Mapping[str, JointPassive] | None = None,
    actuator_forcerange: T.Mapping[str, tuple[float, float]] | None = None,
    joint_range: T.Mapping[str, tuple[float, float]] | None = None,
  ):
    """Initializes the body.

    Args:
      observable_proprio: Whether the proprioceptive observations are included.
      observable_tactile: Whether the foot tactile observations are included.
      observable_vestibular: Whether the head orientation, angular velocity and acceleration are
        included.
      observable_visual: Unsupported (raises if true).
      actionable_dofs: Actuators that the policy controls, by segment. Other actuators get only
        passive torques.
      contact_mode: Which geoms collide with the ground: 'foot' (feet and tail), 'trunk-limbs'
        (feet, trunk and upper limbs), 'trunk-limbs-tail' (and the tail), 'trunk-limbs-tail-head'
        (and the head), 'no-self' (every collision geom, not with each other), or 'all' (the XML's).
        The trunk, upper limb and head geoms are the termination contact geoms.
      foot_compliance: Contact parameters (solref, solimp, condim, margin) of the foot geoms.
      delete_geoms: Whether to delete the geoms that do not collide.
      rescale_mode: How positions are rescaled: 'limits' (to [0, 1] over the position ranges) or
        'setpoint_delta' (to [-1, 1] around `RAT_ACTUATORS_POS_MID`).
      rescale_observations: Whether the position observations are rescaled.
      rescale_actions: Whether the position (and stiffness) actions are rescaled.
      noise_observations: Unused.
      noise_actions: Unused.
      controller: 'position' or 'position-stiffness' (PD controllers of the actionable actuators),
        a transform, a mapping of transforms by segment, or `None` for torques only.
      controller_actkey_prefixes: Action key prefixes of a transform controller, reduced to the
        actionable actuators.
      controller_obskey_prefixes: Observation key prefixes of a transform controller, reduced to
        the actionable actuators.
      actionable_trq: Whether the torques of the actionable actuators are actions.
      transforms: Further transforms, after the body's.
      passive_kp: Passive stiffness per actuator, by segment.
      passive_kd: Passive damping per actuator, by segment.
      root_pos_init: Initial root position (m), [3].
      root_quat_init: Initial root orientation (w, x, y, z), [4].
      joints_pos_init: Initial joint positions (rad, without the root), [55].
      xml_path: Path of the model XML.
      head_global_linvel_sensor: Whether to add the head's global linear velocity sensor.
      joint_passive: Overrides of joints' passive properties (see below).
      actuator_forcerange: Overrides of actuators' torque limits (see below).
      joint_range: Overrides of joints' ranges (see below).

    Raises:
      ValueError: For an invalid `contact_mode` or `rescale_mode`.
      TypeError: For an invalid `controller`.
    """
    self._observable_proprio = observable_proprio
    self._observable_tactile = observable_tactile
    self._observable_vestibular = observable_vestibular
    self._observable_visual = observable_visual

    # Load spec from XML file and cache model elements.
    mj_spec = mj.MjSpec.from_file(str(xml_path))
    assert mj_spec.compiler.degree == True

    # Overrides of the XML, applied before the model elements and spaces below are built.
    # `joint_passive` overrides the passive properties (stiffness, damping, armature) of joints,
    # keyed by joint name or joint class name (a name key wins over a class key), e.g.
    # {'hip_L_extend': {'damping': 0.002}, 'cervical_extend': {'stiffness': 0.02}}.
    # Unset keys keep the XML values. Note the limb XML damping equals `passive_kd` by default.
    if joint_passive:
      names = {j.name for j in mj_spec.joints} | {j.classname.name for j in mj_spec.joints}
      assert set(joint_passive) <= names, (
        f'Invalid `joint_passive` keys: {set(joint_passive) - names}'
      )
      for joint in mj_spec.joints:
        params = joint_passive.get(joint.name, joint_passive.get(joint.classname.name))
        if params is None: continue
        assert set(params) <= JointPassive.__optional_keys__, (
          f'Invalid `joint_passive` params: {set(params)}'
        )
        if 'stiffness' in params: joint.stiffness[0] = params['stiffness']
        if 'damping' in params: joint.damping[0] = params['damping']
        if 'armature' in params: joint.armature = params['armature']

    # `actuator_forcerange` overrides the torque limits [low, high] (N.m, or N for tendons) of
    # actuators by name, e.g. {'hip_L_extend': [-0.3, 0.7]}. Asymmetric limits cap the net flexor
    # (negative) and extensor (positive) torques separately; they also set the ctrl and action
    # ranges.
    if actuator_forcerange:
      names = {a.name for a in mj_spec.actuators}
      assert set(actuator_forcerange) <= names, (
        f'Invalid `actuator_forcerange` keys: {set(actuator_forcerange) - names}'
      )
      for actuator in mj_spec.actuators:
        if actuator.name not in actuator_forcerange: continue
        low, high = actuator_forcerange[actuator.name]
        assert low < 0 < high, f'Invalid `actuator_forcerange` for {actuator.name}: {low, high}'
        actuator.forcerange = (low, high)

    # `joint_range` overrides the range [low, high] (deg) of joints by name, e.g.
    # {'ankle_L': [-15, 80]}. The range also sets the proprio_pos box, i.e. the spring-muscle
    # low/high.
    if joint_range:
      names = {j.name for j in mj_spec.joints}
      assert set(joint_range) <= names, f'Invalid `joint_range` keys: {set(joint_range) - names}'
      for joint in mj_spec.joints:
        if joint.name not in joint_range: continue
        low, high = joint_range[joint.name]
        assert low < high, f'Invalid `joint_range` for {joint.name}: {low, high}'
        joint.range = (low, high)

    self._actuators = actuators = mj_spec.actuators
    assert len(self._actuators) == 38
    assert all(a.trntype in (mj.mjtTrn.mjTRN_JOINT, mj.mjtTrn.mjTRN_TENDON) for a in actuators)
    self._actuateds = [
      mj_spec.joint(a.target) if a.trntype == mj.mjtTrn.mjTRN_JOINT else mj_spec.tendon(a.target)
      for a in actuators
    ]
    self._actuator_names: tuple[str] = tuple(a.name for a in actuators)

    self._root_body = mj_spec.body('torso')
    self._pelvis_body = mj_spec.body('pelvis')
    self._root_joint = mj_spec.joint('torso')
    self._root_local_linvel = mj_spec.sensor('local_linvel')
    self._root_global_linvel = mj_spec.sensor('global_linvel')
    self._root_global_angvel = mj_spec.sensor('global_angvel')
    self._root_global_linacc = mj_spec.sensor('global_linacc')

    self._head_quaternion = mj_spec.sensor('quaternion')
    self._head_gyroscope = mj_spec.sensor('gyroscope')
    self._head_accelerometer = mj_spec.sensor('accelerometer')
    self._head_global_linvel = mj_spec.add_sensor(
      name='head_global_linvel',
      type=mj.mjtSensor.mjSENS_FRAMELINVEL,
      objtype=mj.mjtObj.mjOBJ_SITE,
      objname='head',
    ) if head_global_linvel_sensor else None

    self._foot_names = ('FL', 'FR', 'HL', 'HR')
    self._foot_touch = [
      mj_spec.sensor(f'{x}_touch_{i}') for x in self._foot_names for i in ('digit', 'paw')
    ]
    self._foot_global_linvel = [mj_spec.sensor(f'{x}_global_linvel') for x in self._foot_names]
    self._foot_global_pos = [mj_spec.sensor(f'{x}_global_pos') for x in self._foot_names]

    # Foot contact forces are computed from the contact data (`get_foot_contact_forces`) rather than
    # with sensors: MJX implements neither per-body nor net-force contact sensors, and per-geom
    # sensors give forces in the contact frame, not the world frame.

    # Set all actuator types to direct drive.
    for actuator in actuators:
      actuator.dyntype = mj.mjtDyn.mjDYN_NONE
      actuator.ctrlrange = actuator.forcerange
      actuator.gainprm[:3] = (1., 0., 0.)
      actuator.biasprm[:3] = (0., 0., 0.)

    # Set contact mode of collision geoms.
    match contact_mode:
      case 'foot':
        self._termination_contact_geoms = []
        # Tail collision.
        names = [
          'vertebrae_C16C20_collision',
          'vertebrae_C26C30_collision',
        ]
        for geom in mj_spec.geoms:
          if geom.classname.name == 'visual': continue
          geom.contype = 1 if geom.classname.name == 'collision_foot' or geom.name in names else 0
          geom.conaffinity = 0
      case 'trunk-limbs':
        # Feet, trunk, and upper limbs collide; the tail does not, so it cannot prop up the body.
        self._termination_contact_geoms = names = [
          'torso_collision',
          'pelvis_gut_collision',
          'humerus_L_collision',
          'humerus_R_collision',
          'upper_leg_L_collision',
          'upper_leg_R_collision',
        ]
        for geom in mj_spec.geoms:
          if geom.classname.name == 'visual': continue
          geom.contype = 1 if geom.classname.name == 'collision_foot' or geom.name in names else 0
          geom.conaffinity = 0
      case 'trunk-limbs-tail':
        # `trunk-limbs` with tail collision.
        self._termination_contact_geoms = names = [
          'torso_collision',
          'pelvis_gut_collision',
          'humerus_L_collision',
          'humerus_R_collision',
          'upper_leg_L_collision',
          'upper_leg_R_collision',
        ]
        # Tail collision.
        names = names + [
          'vertebrae_C16C20_collision',
          'vertebrae_C26C30_collision',
        ]
        for geom in mj_spec.geoms:
          if geom.classname.name == 'visual': continue
          geom.contype = 1 if geom.classname.name == 'collision_foot' or geom.name in names else 0
          geom.conaffinity = 0
      case 'trunk-limbs-tail-head':
        # `trunk-limbs-tail`, with the head (cranium, snout, jaw, chin) also colliding and
        # terminating on contact.
        self._termination_contact_geoms = names = [
          'torso_collision',
          'pelvis_gut_collision',
          'humerus_L_collision',
          'humerus_R_collision',
          'upper_leg_L_collision',
          'upper_leg_R_collision',
          'skull_0_collision',
          'skull_F_collision',
          'jaw_back_collision',
          'jaw_F_collision',
        ]
        # Tail collision.
        names = names + [
          'vertebrae_C16C20_collision',
          'vertebrae_C26C30_collision',
        ]
        for geom in mj_spec.geoms:
          if geom.classname.name == 'visual': continue
          geom.contype = 1 if geom.classname.name == 'collision_foot' or geom.name in names else 0
          geom.conaffinity = 0
      case 'no-self':
        self._termination_contact_geoms = []
        for geom in mj_spec.geoms:
          if geom.classname.name == 'visual': continue
          geom.contype = 1
          geom.conaffinity = 0
      case 'all':
        self._termination_contact_geoms = []
        pass
      case _:
        raise ValueError(f'Invalid `contact_mode`: {contact_mode}')

    # Set contact compliance (solref/solimp/condim/margin) of foot geoms, e.g. to make foot contacts
    # softer/grippier instead of rigid and bouncy. Applies uniformly to all `collision_foot` geoms
    # (both fore and hind feet).
    if foot_compliance is not None:
      for geom in mj_spec.geoms:
        if geom.classname.name != 'collision_foot': continue
        if 'solref' in foot_compliance: geom.solref = foot_compliance['solref']
        if 'solimp' in foot_compliance: geom.solimp = foot_compliance['solimp']
        if 'condim' in foot_compliance: geom.condim = foot_compliance['condim']
        if 'margin' in foot_compliance: geom.margin = foot_compliance['margin']

    # Delete all geoms not used for collision for computational efficiency.
    if delete_geoms:
      for geom in mj_spec.geoms:
        if geom.contype == 0:
          mj_spec.delete(geom)

    # Identify observation/action constraints.
    assert all(k in RAT_ACTUATORS for k in actionable_dofs), (
      f'Invalid `actionable_dofs` keys: {set(actionable_dofs) - set(RAT_ACTUATORS)} '
      f'vs {set(RAT_ACTUATORS)}.'
    )
    actionable_dofs = {k: actionable_dofs.get(k, ()) for k in RAT_ACTUATORS}
    self._actionable_dofs = actionable_dofs

    # Set passive forces.
    passive_transforms: list[cx.Transform] = []
    for segment in RAT_ACTUATORS:
      passive_transforms.append(
        cxt.ProportionalDerivativeController(
          actionable_pos=None,
          actionable_vel=None,
          actionable_kp=None,
          actionable_kd=None,
          actionable_trq=None,
          pos_default=RAT_ACTUATORS_POS_MID[segment],
          kp=passive_kp[segment],
          kd=passive_kd[segment],
          obskey_pos=f'proprio_pos_{segment}',
          obskey_vel=f'proprio_vel_{segment}',
          actkey_trq=f'actuators_trq_{segment}',
          additive_trq=True,
        )
      )

    # Prevent uncontrolled actuators from being actuated by overriding their torques to zero.
    override_action_transform = cxt.OverrideAction({
      f'actuators_trq_{segment}': [
        None if name in actionable_dofs[segment] else 0.0 for name in segment_actuators
      ]
      for segment, segment_actuators in RAT_ACTUATORS.items() if actionable_dofs[segment]
    })

    # Set controller transforms.
    controller_transforms: list[cx.Transform] = []
    match controller:
      case 'position':
        controller_actkey_prefixes = ['actuators_pos']
        controller_obskey_prefixes = []
        for segment in RAT_ACTUATORS:
          if not actionable_dofs[segment]: continue
          transform = cxt.ProportionalDerivativeController(
            actionable_pos=f'actuators_pos_{segment}',
            actionable_kp=None,
            kp=RAT_ACTUATORS_KP[segment],
            kd=RAT_ACTUATORS_KD[segment],
            obskey_pos=f'proprio_pos_{segment}',
            obskey_vel=f'proprio_vel_{segment}',
            actkey_trq=f'actuators_trq_{segment}',
            additive_trq=True,
          )
          controller_transforms.append(transform)
      case 'position-stiffness':
        controller_actkey_prefixes = ['actuators_pos', 'actuators_kp']
        controller_obskey_prefixes = []
        for segment in RAT_ACTUATORS:
          if not actionable_dofs[segment]: continue
          transform = cxt.ProportionalDerivativeController(
            actionable_pos=f'actuators_pos_{segment}',
            actionable_kp=f'actuators_kp_{segment}',
            kp=RAT_ACTUATORS_KP[segment],
            kd=RAT_ACTUATORS_KD[segment],
            obskey_pos=f'proprio_pos_{segment}',
            obskey_vel=f'proprio_vel_{segment}',
            actkey_trq=f'actuators_trq_{segment}',
            additive_trq=True,
          )
          controller_transforms.append(transform)
      case None:
        pass
      case cx.Transform():
        controller_transforms.append(controller)
      case collections.abc.Mapping():
        controller_transforms.extend(controller.values())
      case _:
        raise TypeError(f'Invalid `controller`: {controller!r}')

    # Set observation/action rescale transforms.
    rescale_transforms: list[cx.Transform] = []
    match rescale_mode:
      case 'setpoint_delta':
        obs_min, obs_mid, obs_max = {}, {}, {}
        act_min, act_mid, act_max, act_exclude = {}, {}, {}, []
        for segment in RAT_ACTUATORS:
          if not actionable_dofs[segment]: continue
          obs_min[f'proprio_pos_{segment}'] = RAT_ACTUATORS_POS_MIN[segment]
          obs_mid[f'proprio_pos_{segment}'] = RAT_ACTUATORS_POS_MID[segment]
          obs_max[f'proprio_pos_{segment}'] = RAT_ACTUATORS_POS_MAX[segment]
          act_min[f'actuators_pos_{segment}'] = RAT_ACTUATORS_POS_MIN[segment]
          act_min[f'actuators_kp_{segment}'] = 0.
          act_mid[f'actuators_pos_{segment}'] = RAT_ACTUATORS_POS_MID[segment]
          act_mid[f'actuators_kp_{segment}'] = 0.
          act_max[f'actuators_pos_{segment}'] = RAT_ACTUATORS_POS_MAX[segment]
          act_max[f'actuators_kp_{segment}'] = RAT_ACTUATORS_KP[segment]
          act_exclude += [f'muscle_flx_{segment}', f'muscle_ext_{segment}']
        if rescale_observations and obs_min:
          rescale_transforms.append(
            cxt.RescaleObservation(min=obs_min, mid=obs_mid, max=obs_max, signed=True)
          )
        if rescale_actions and act_min:
          rescale_transforms.append(cxt.RescaleAction(
            min=act_min, mid=act_mid, max=act_max, signed=True, exclude=act_exclude,
          ))
      case 'limits':
        obs_scale, obs_signed = {}, {}
        act_scale, act_signed, act_exclude = {}, {}, []
        for segment in RAT_ACTUATORS:
          if not actionable_dofs[segment]: continue
          obs_scale[f'proprio_pos_{segment}'] = (0., 1.)
          obs_signed[f'proprio_pos_{segment}'] = False
          act_scale[f'actuators_pos_{segment}'] = (0., 1.)
          act_scale[f'actuators_kp_{segment}'] = (0., 1.)
          act_signed[f'actuators_pos_{segment}'] = False
          act_signed[f'actuators_kp_{segment}'] = False
          act_exclude += [f'muscle_flx_{segment}', f'muscle_ext_{segment}']
        if rescale_observations and obs_scale:
          rescale_transforms.append(cxt.RescaleObservation(scale=obs_scale, signed=obs_signed))
        if rescale_actions and act_scale:
          rescale_transforms.append(
            cxt.RescaleAction(scale=act_scale, signed=act_signed, exclude=act_exclude)
          )
      case _:
        raise ValueError(f'Invalid `rescale_mode`: {rescale_mode}')

    # Mask the Rat observations and actions to only include actively controlled actuators.
    reduce_observation: dict[str, list[int]] = {}
    reduce_action: dict[str, list[float | None]] = {}
    for segment, segment_actuators in RAT_ACTUATORS.items():
      # Remove uncontrolled observations from model view.
      indices = [segment_actuators.index(name) for name in actionable_dofs[segment]]
      reduce_observation[f'proprio_pos_{segment}'] = indices
      reduce_observation[f'proprio_vel_{segment}'] = indices
      reduce_observation[f'proprio_trq_{segment}'] = indices
      for obskey_prefix in controller_obskey_prefixes:
        reduce_observation[f'{obskey_prefix}_{segment}'] = indices
      # Remove uncontrolled actions from model view. The fallback actions can be zero because
      # the controller output torques will be ignored, anyway.
      act_mask = [None if name in actionable_dofs[segment] else 0.0 for name in segment_actuators]
      if any(x is not None for x in act_mask):
        for actkey_prefix in controller_actkey_prefixes:
          reduce_action[f'{actkey_prefix}_{segment}'] = act_mask
      # Hide torques from the model view unless they are directly actionable, in which case only
      # uncontrolled torques are hidden (and zeroed).
      reduce_action[f'actuators_trq_{segment}'] = (
        act_mask if actionable_trq else [0] * len(segment_actuators)
      )
    interface_transforms: list[cx.Transform] = [
      cxt.ReduceObservation(reduce_observation),
      cxt.ReduceAction(reduce_action),
    ]

    # Define initial pos for each episode.
    self._root_pos_init = jnp.asarray(root_pos_init, dtype=float)
    assert self._root_pos_init.shape == (3,)
    self._root_quat_init = jnp.asarray(root_quat_init, dtype=float)
    assert self._root_quat_init.shape == (4,)
    self._joints_pos_init = jnp.asarray(joints_pos_init, dtype=float)
    assert self._joints_pos_init.shape == (55,)

    super().__init__(
      transforms=[
        # Substep transforms.
        cxt.ClipAction(),
        # NoisyObservation(noise_observations) if noise_observations else None,
        # NoisyAction(noise_actions) if noise_actions else None,
        *passive_transforms,        # Passive forces.
        override_action_transform,  # Override uncontrolled actuators to zero torque.
        *controller_transforms,     # Active controller transforms.
        *rescale_transforms,        # Rescale observations and actions.
        *interface_transforms,      # Mask out observations and actions for uncontrolled actuators.
        cxt.ClipAction(),
        # Step transforms.
        # cxt.BufferObservationSubsteps(),
        # cxt.BufferActionSubsteps(),
        *transforms,
      ],
      mj_spec=mj_spec,
      local_state=RatState(
        foot_contact=jnp.zeros(4, dtype=bool),
        foot_liftoff=jnp.zeros(4, dtype=bool),
        foot_liftoff_time=jnp.zeros(4),
        foot_liftoff_pos=jnp.zeros((4, 3)),
        foot_touchdown=jnp.zeros(4, dtype=bool),
        foot_touchdown_time=jnp.zeros(4),
        foot_stance_duration=jnp.zeros(4),
        foot_swing_duration=jnp.zeros(4),
        foot_swing_length=jnp.zeros(4),
        foot_swing_peak=jnp.zeros(4),
      ),
    )

  @T.override
  def initialize(
    self,
    state: cx.MjxEnvState,
    context: cx.Context,
  ) -> tuple[cx.MjxEnvState, cx.Context]:
    """Passes the actuators to the transforms (`context['actuators']`)."""
    context['actuators'] = self._actuators
    return state, context

  @T.override
  def action_space(self, state: cx.MjxEnvState) -> cx.Dict:
    """Returns the torque action space per segment, `actuators_trq_<segment>`, bounded by the
    actuators' control ranges."""
    space = cx.Dict()
    low = jnp.array([state.mjx_model.bind(a).ctrlrange[0] for a in self._actuators])
    high = jnp.array([state.mjx_model.bind(a).ctrlrange[1] for a in self._actuators])
    for segment, actuators in RAT_ACTUATORS.items():
      indices = [RAT_ACTUATOR_NAMES.index(a) for a in actuators]
      labels = [[self._actuator_names[i] for i in indices]]
      indices = jnp.array(indices)
      space[f'actuators_trq_{segment}'] = cx.Box(
        low=low[indices],
        high=high[indices],
        name=f'actuators_trq_{segment}',
        desc=f'Actuator output torques (N.m) for {segment}',
        labels=labels,
      )
    return space

  @T.override
  def action(self, state: cx.MjxEnvState, action: cx.ArrayDict) -> cx.MjxEnvState:
    """Sets the actuators' controls to the torques, in the model's actuator order."""
    actuators = state.mjx_data.bind(state.mjx_model, self._actuators)
    actuators_trq = jnp.concat(
      [action[f'actuators_trq_{segment}'] for segment in RAT_ACTUATORS.keys()]
    )
    mjx_data = actuators.set('ctrl', actuators_trq)
    return state.replace(mjx_data=mjx_data)

  @T.override
  def observation_space(self, state: cx.MjxEnvState) -> cx.ObservationSpace:
    """
    Actuators are ordered rostral-to-caudal, left-to-right, and proximal-to-distal:
    ```
    [
      mandible, atlas,                                                     # head
      cervical_extend, cervical_bend, cervical_twist,                      # cervical
      scapula_L_supinate, scapula_L_abduct, scapula_L_extend,              # arm L
      shoulder_L_supinate, shoulder_L_extend, elbow_L, wrist_L, finger_L,  # arm L
      scapula_R_supinate, scapula_R_abduct, scapula_R_extend,              # arm R
      shoulder_R_supinate, shoulder_R_extend, elbow_R, wrist_R, finger_R,  # arm R
      lumbar_extend, lumbar_bend, lumbar_twist,                            # lumbar
      hip_L_supinate, hip_L_abduct, hip_L_extend, knee_L, ankle_L, toe_L,  # leg L
      hip_R_supinate, hip_R_abduct, hip_R_extend, knee_R, ankle_R, toe_R,  # leg R
      tail_extend, tail_bend,                                              # tail
    ]
    ```

    Positions are joint angles (rad), and lengths (m) for the tendons (cervical, lumbar and tail).
    """
    space = cx.Dict()
    if self._observable_proprio:
      limits = jnp.asarray([
        np.deg2rad(a.range) if isinstance(a, mj.MjsJoint) else a.range for a in self._actuateds
      ])  # [38, 2], (rad)
      # Ungrouped actuator spaces.
      boxes = dict(
        proprio_pos=cx.Box(
          low=limits[:, 0],
          high=limits[:, 1],
          name='proprio_pos',
          desc='Proprioceptive actuator angular positions (rad)',
        ),
        proprio_vel=cx.Box(
          low=-RAT_VEL_MAX * jnp.ones(len(self._actuators)),
          high=+RAT_VEL_MAX * jnp.ones(len(self._actuators)),
          name='proprio_vel',
          desc='Proprioceptive actuator angular velocities (rad/s)',
        ),
        proprio_trq=cx.Box(
          low=-RAT_TRQ_MAX * jnp.ones(len(self._actuators)),
          high=+RAT_TRQ_MAX * jnp.ones(len(self._actuators)),
          name='proprio_trq',
          desc='Proprioceptive actuator output torques (N.m) and forces (N)',
        ),
      )
      # Split ungrouped actuator spaces into grouped segment spaces.
      for key, box in boxes.items():
        for segment, actuators in RAT_ACTUATORS.items():
          indices = [RAT_ACTUATOR_NAMES.index(a) for a in actuators]
          labels = [[self._actuator_names[i] for i in indices]]
          indices = jnp.array(indices)
          name = f'{key}_{segment}'
          space[name] = cx.Box(
            low=box.low[indices],
            high=box.high[indices],
            name=name,
            desc=box.desc + f' for {segment}',
            labels=labels,
          )
    if self._observable_tactile:
      for foot_name in self._foot_names:
        labels = [[f'{foot_name}_{i}' for i in ('digit', 'paw')]]
        space[f'tactile_limb{foot_name}'] = cx.Box(
          low=jnp.zeros(2),
          high=RAT_FOOT_MAX * jnp.ones(2),
          name=f'tactile_limb{foot_name}:[digit,paw]',
          desc=f'Tactile contact forces (N) for {foot_name}',
          labels=labels,
        )
    if self._observable_vestibular:
      space['vestibular_rpy'] = cx.Box(
        low=-jnp.pi * jnp.ones(3),
        high=+jnp.pi * jnp.ones(3),
        name='vestibular_rpy:[roll,pitch,yaw]',
        desc='Vestibular orientation, euler angles (rad)',
      )
      space['vestibular_angvel'] = cx.Box(
        low=-RAT_ANGVEL_MAX * jnp.ones(3),
        high=+RAT_ANGVEL_MAX * jnp.ones(3),
        name='vestibular_angvel:[droll,dpitch,dyaw]',
        desc='Vestibular angular velocity (rad/s)',
      )
      space['vestibular_linacc'] = cx.Box(
        low=-RAT_LINACC_MAX * jnp.ones(3),
        high=+RAT_LINACC_MAX * jnp.ones(3),
        name='vestibular_linacc:[ddx,ddy,ddz]',
        desc='Vestibular linear acceleration (m/s^2)',
      )
    if self._observable_visual:
      raise NotImplementedError()
    return space

  @T.override
  def observation(self, state: cx.MjxEnvState) -> tuple[cx.ArrayDict, cx.MjxEnvState]:
    """Returns the observations (see `observation_space`)."""
    observation = collections.OrderedDict()
    if self._observable_proprio:
      for key, value in dict(
        proprio_pos=self.get_proprio_pos(state),
        proprio_vel=self.get_proprio_vel(state),
        proprio_trq=self.get_proprio_trq(state),
      ).items():
        for segment, actuators in RAT_ACTUATORS.items():
          indices = jnp.array([RAT_ACTUATOR_NAMES.index(a) for a in actuators])
          observation[f'{key}_{segment}'] = value[indices]
    if self._observable_tactile:
      tactile_foot = self.get_tactile_foot(state)
      for foot_name, tactile_data in zip(self._foot_names, jnp.split(tactile_foot, 4)):
        observation[f'tactile_limb{foot_name}'] = tactile_data
    if self._observable_vestibular:
      observation['vestibular_rpy'] = self.get_sensor_rpy(state)
      observation['vestibular_angvel'] = self.get_sensor_angvel(state)
      observation['vestibular_linacc'] = self.get_sensor_linacc(state)
    if self._observable_visual:
      raise NotImplementedError()
    return observation, state

  # ------------------------------------------------------------------------------------------------
  # Getters (observable properties)

  def get_actuators(self) -> tuple[str, ...]:
    """Returns the actuator names, in the model's order."""
    return self._actuator_names

  def get_termination_contact_geoms(self) -> list[str]:
    """Returns the geoms whose ground contact ends an episode (see `contact_mode`)."""
    return self._termination_contact_geoms

  # ------------------------------------------------------------------------------------------------
  # Getters (observable data)

  def get_proprio_pos(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the actuated joints' angles (rad) and tendons' lengths (m), [38]."""
    jidxs, js = zip(*[(i, x) for i, x in enumerate(self._actuateds) if isinstance(x, mj.MjsJoint)])
    tidxs, ts = zip(*[(i, x) for i, x in enumerate(self._actuateds) if isinstance(x, mj.MjsTendon)])
    pos = jnp.zeros(len(self._actuateds))
    pos = pos.at[jnp.array(jidxs)].set(state.mjx_data.bind(state.mjx_model, js).qpos)  # [30]
    pos = pos.at[jnp.array(tidxs)].set(state.mjx_data.bind(state.mjx_model, ts).length)  # [8]
    return pos  # [38]

  def get_proprio_vel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the actuated joints' (rad/s) and tendons' (m/s) velocities, [38]."""
    jidxs, js = zip(*[(i, x) for i, x in enumerate(self._actuateds) if isinstance(x, mj.MjsJoint)])
    tidxs, ts = zip(*[(i, x) for i, x in enumerate(self._actuateds) if isinstance(x, mj.MjsTendon)])
    vel = jnp.zeros(len(self._actuateds))
    vel = vel.at[jnp.array(jidxs)].set(state.mjx_data.bind(state.mjx_model, js).qvel)  # [30]
    vel = vel.at[jnp.array(tidxs)].set(state.mjx_data.bind(state.mjx_model, ts).velocity)  # [8]
    return vel  # [38]

  def get_proprio_trq(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the actuator forces (N.m, or N for tendons), [38]."""
    return state.mjx_data.bind(state.mjx_model, self._actuators).force  # [38]

  def get_proprio_passive(self, state: cx.MjxEnvState) -> jax.Array:
    """MuJoCo passive force (joint stiffness and damping from the XML) on the actuated joints,
    NaN for tendons. The body's own passive torques (`passive_kp`, `passive_kd`) are part of the
    actuator force (`get_proprio_trq`) instead."""
    jidxs, js = zip(*[(i, x) for i, x in enumerate(self._actuateds) if isinstance(x, mj.MjsJoint)])
    passive = jnp.full(len(self._actuateds), jnp.nan)
    qfrc_passive = state.mjx_data.bind(state.mjx_model, js).qfrc_passive  # [30]
    passive = passive.at[jnp.array(jidxs)].set(qfrc_passive)
    return passive  # [38]

  def get_tactile_foot(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the touch sensor forces (N) of each foot's digit and paw, (FL, FR, HL, HR), [8]."""
    return state.mjx_data.bind(state.mjx_model, self._foot_touch).sensordata[:8]  # [8]

  # def get_tactile_foot_force(self, state: cx.MjxEnvState) -> jax.Array:
  #   return state.mjx_data.bind(state.mjx_model, self._foot_force).sensordata.reshape(8, 3)

  def get_sensor_quat(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the head's global orientation (w, x, y, z), [4]."""
    return state.mjx_data.bind(state.mjx_model, self._head_quaternion).sensordata[:4]  # [4]

  def get_sensor_rpy(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the head's global roll, pitch and yaw (rad), [3]."""
    return cxm.quat_to_rpy(self.get_sensor_quat(state))  # [3]

  def get_sensor_gravity(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the gravity direction in the head frame, [3]."""
    return cxm.quat_to_xmat(self.get_sensor_quat(state)).T @ jnp.array([0, 0, -1])  # [3]

  def get_sensor_angvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the head's angular velocity (rad/s, head frame), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._head_gyroscope).sensordata[:3]  # [3]

  def get_sensor_linacc(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the head's linear acceleration (m/s^2, head frame), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._head_accelerometer).sensordata[:3]  # [3]

  # ------------------------------------------------------------------------------------------------
  # Getters (privledged data)

  def get_pose_init(self) -> jax.Array:
    """Returns the initial positions (rad) of the `POSE_JOINTS`."""
    joints = [j for i, j in enumerate(self.mj_spec.joints) if j.name != 'torso']
    pose_indices = [i for i, j in enumerate(joints) if j.name in POSE_JOINTS]
    return self._joints_pos_init[jnp.array(pose_indices)]

  def get_pose(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the positions (rad) of the `POSE_JOINTS`."""
    joints = [j for j in self.mj_spec.joints if j.name in POSE_JOINTS]  # Exclude root joint.
    return state.mjx_data.bind(state.mjx_model, joints).qpos

  def get_root_local_linvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the linear velocity (m/s) of the `local_linvel` velocimeter (on the head site), in
    its frame, [3]."""
    return state.mjx_data.bind(state.mjx_model, self._root_local_linvel).sensordata[:3]  # [3]

  def get_root_global_linvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root's global linear velocity (m/s), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._root_global_linvel).sensordata[:3]  # [3]

  def get_root_global_angvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root's global angular velocity (rad/s), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._root_global_angvel).sensordata[:3]  # [3]

  def get_root_global_linacc(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root's global linear acceleration (m/s^2), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._root_global_linacc).sensordata[:3]  # [3]

  def get_head_global_linvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Global linear velocity (m/s) of the head site; needs `head_global_linvel_sensor=True`."""
    assert self._head_global_linvel is not None, 'Requires `head_global_linvel_sensor=True`.'
    return state.mjx_data.bind(state.mjx_model, self._head_global_linvel).sensordata[:3]  # [3]

  def get_root_global_pos(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root (torso) body's global position (m), [3]."""
    return state.mjx_data.bind(state.mjx_model, self._root_body).xpos[:3]  # [3]

  def get_root_global_quat(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root (torso) body's global orientation (w, x, y, z), [4]."""
    return state.mjx_data.bind(state.mjx_model, self._root_body).xquat[:4]  # [4]

  def get_pelvis_global_quat(self, state: cx.MjxEnvState) -> jax.Array:
    """Global orientation quaternion (w, x, y, z) of the pelvis body."""
    return state.mjx_data.bind(state.mjx_model, self._pelvis_body).xquat[:4]  # [4]

  def get_root_global_rpy(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the root (torso) body's global roll, pitch and yaw (rad), [3]."""
    return cxm.quat_to_rpy(self.get_root_global_quat(state))  # [3]

  def get_foot_global_linvel(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns each foot's global linear velocity (m/s), (FL, FR, HL, HR), [4, 3]."""
    return state.mjx_data.bind(state.mjx_model,
                               self._foot_global_linvel).sensordata.reshape(4, 3)  # [4, 3]

  def get_foot_global_pos(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns each foot's global position (m), (FL, FR, HL, HR), [4, 3]."""
    return state.mjx_data.bind(state.mjx_model,
                               self._foot_global_pos).sensordata.reshape(4, 3)  # [4, 3]

  def get_foot_global_height(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns each foot's global height (m), (FL, FR, HL, HR), [4]."""
    return state.mjx_data.bind(state.mjx_model,
                               self._foot_global_pos).sensordata.reshape(4, 3)[:, 2]  # [4]

  def get_foot_contact_forces(self, state: cx.MjxEnvState) -> jax.Array:
    """Returns the global contact forces (N) on the foot bodies (hand and finger of the forelimbs,
    foot and toe of the hindlimbs; L then R), [8, 3]."""
    foot_body_ids = [
      mj.mj_name2id(state.mj_model, mj.mjtObj.mjOBJ_BODY.value, x)
      for x in ('hand_L', 'finger_L', 'hand_R', 'finger_R', 'foot_L', 'toe_L', 'foot_R', 'toe_R')
    ]
    return cx.contact_force_body_world(state.mjx_model, state.mjx_data, foot_body_ids)  # [8, 3]

  # ------------------------------------------------------------------------------------------------
  # Setters

  def set_root_pos_rpy(
    self, state: cx.MjxEnvState, pos: jax.Array, rpy: jax.Array
  ) -> cx.MjxEnvState:
    """Set the root body position and orientation using roll-pitch-yaw angles."""
    quat = cxm.rpy_to_quat(rpy)
    root_joint = state.mjx_data.bind(state.mjx_model, self._root_joint)
    mjx_data = root_joint.set('qpos', jnp.concatenate([pos, quat]))
    return state.replace(mjx_data=mjx_data)

  def set_root_pos_quat(
    self, state: cx.MjxEnvState, pos: jax.Array, quat: jax.Array
  ) -> cx.MjxEnvState:
    """Set the root body position and orientation using a quaternion."""
    root_joint = state.mjx_data.bind(state.mjx_model, self._root_joint)
    mjx_data = root_joint.set('qpos', jnp.concatenate([pos, quat]))
    return state.replace(mjx_data=mjx_data)

  def set_root_vel(
    self, state: cx.MjxEnvState, linvel: jax.Array, angvel: jax.Array
  ) -> cx.MjxEnvState:
    """Set the root body linear velocity [vx, vy, vz] and angular velocity [wx, wy, wz]."""
    root_joint = state.mjx_data.bind(state.mjx_model, self._root_joint)
    mjx_data = root_joint.set('qvel', jnp.concatenate([linvel, angvel]))
    return state.replace(mjx_data=mjx_data)

  def set_root_perturbation(
    self, state: cx.MjxEnvState, xfrc: jax.Array, xtrq: jax.Array
  ) -> cx.MjxEnvState:
    """Apply a perturbation force [x, y, z] and torque [x, y, z] to the root body."""
    root_body = state.mjx_data.bind(state.mjx_model, self._root_body)
    mjx_data = root_body.set('xfrc_applied', jnp.concatenate([xfrc, xtrq]))
    return state.replace(mjx_data=mjx_data)

  def set_joints_pos(self, state: cx.MjxEnvState, pos: jax.Array) -> cx.MjxEnvState:
    """Set the joint positions (excluding root)."""
    joints = [j for j in self.mj_spec.joints if j.name != 'torso']  # Exclude root joint.
    mjx_data = state.mjx_data.bind(state.mjx_model, joints).set('qpos', pos)
    return state.replace(mjx_data=mjx_data)

  def set_joints_vel(self, state: cx.MjxEnvState, vel: jax.Array) -> cx.MjxEnvState:
    """Set the joint velocities (excluding root)."""
    joints = [j for j in self.mj_spec.joints if j.name != 'torso']  # Exclude root joint.
    mjx_data = state.mjx_data.bind(state.mjx_model, joints).set('qvel', vel)
    return state.replace(mjx_data=mjx_data)

  # ------------------------------------------------------------------------------------------------
  # Initializers

  def init_pose(self, state: cx.MjxEnvState) -> cx.MjxEnvState:
    """Set the initial positions and velocities."""
    state = self.set_root_pos_quat(state, self._root_pos_init, self._root_quat_init)
    state = self.set_root_vel(state, jnp.zeros(3), jnp.zeros(3))
    state = self.set_joints_pos(state, self._joints_pos_init)
    state = self.set_joints_vel(state, jnp.zeros_like(self._joints_pos_init))
    return state

  # ------------------------------------------------------------------------------------------------
  # Counters

  def init_foot_counters(self, state: cx.MjxEnvState) -> cx.MjxEnvState:
    """Initialize foot contact and swing/stance counters."""
    local = state.local.get(self.index)
    local = local.replace(
      foot_contact=jnp.zeros(4, dtype=bool),
      foot_liftoff=jnp.zeros(4, dtype=bool),
      foot_liftoff_time=jnp.zeros(4),
      foot_liftoff_pos=jnp.zeros((4, 3)),
      foot_touchdown=jnp.zeros(4, dtype=bool),
      foot_touchdown_time=jnp.zeros(4),
      foot_stance_duration=jnp.zeros(4),
      foot_swing_duration=jnp.zeros(4),
      foot_swing_length=jnp.zeros(4),
      foot_swing_peak=jnp.zeros(4),
    )
    state = state.replace(local=state.local.set(self.index, local))
    return state

  def update_foot_counters(self, state: cx.MjxEnvState) -> cx.MjxEnvState:
    """Update foot contact and swing/stance counters."""
    local = state.local.get(self.index)
    curr_contact = self.get_tactile_foot(state).reshape(4, 2).sum(axis=1) > 0  # [4], bool
    curr_pos = self.get_foot_global_pos(state)  # [4, 3]
    curr_time = state.mjx_data.time.repeat(4)  # [4], float

    # A liftoff/touchdown is detected by a change in contact from the previous timestep. At such
    # transitions, recompute all derived metrics then update the most recent liftoff/touchdown time.

    # Liftoff.
    liftoff = ~curr_contact & local.foot_contact  # [4], bool
    stance_duration = jnp.where(
      liftoff,
      curr_time - local.foot_touchdown_time,
      local.foot_stance_duration,
    )  # [4], float
    liftoff_time = jnp.where(
      liftoff,
      curr_time,
      local.foot_liftoff_time,
    )  # [4], float
    liftoff_pos = jnp.where(
      liftoff[:, None],
      curr_pos,
      local.foot_liftoff_pos,
    )  # [4, 3], float

    # Swing.
    swing_peak = jnp.where(
      liftoff,
      jnp.zeros(4),
      local.foot_swing_peak,
    )  # [4], float
    swing_peak = jnp.where(
      ~curr_contact,
      jnp.maximum(swing_peak, curr_pos[:, 2]),
      swing_peak,
    )  # [4], float

    # Touchdown.
    touchdown = curr_contact & ~local.foot_contact  # [4], bool
    swing_duration = jnp.where(
      touchdown,
      curr_time - local.foot_liftoff_time,
      local.foot_swing_duration,
    )  # [4], float
    swing_length = jnp.where(
      touchdown,
      jnp.linalg.norm(curr_pos[:, :2] - local.foot_liftoff_pos[:, :2], axis=1),
      local.foot_swing_length,
    )  # [4], float
    touchdown_time = jnp.where(touchdown, curr_time, local.foot_touchdown_time)  # [4], float

    local = local.replace(
      foot_contact=curr_contact,
      foot_liftoff=liftoff,
      foot_liftoff_time=liftoff_time,
      foot_liftoff_pos=liftoff_pos,
      foot_touchdown=touchdown,
      foot_touchdown_time=touchdown_time,
      foot_stance_duration=stance_duration,
      foot_swing_duration=swing_duration,
      foot_swing_length=swing_length,
      foot_swing_peak=swing_peak,
    )
    state = state.replace(local=state.local.set(self.index, local))
    return state
