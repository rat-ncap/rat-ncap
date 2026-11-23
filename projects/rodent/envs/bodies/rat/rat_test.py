import typing as T

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
import pytest

import composer as cx
from envs.bodies.rat import rat

LIMB_SEGMENTS: tuple[rat.RatSegment, ...] = ('limbFL', 'limbFR', 'limbHL', 'limbHR')
LUMBAR_SEGMENTS: tuple[rat.RatSegment, ...] = ('lumbarE', 'lumbarBT')
SEGMENT_SIZES = dict(limbFL=8, limbFR=8, limbHL=6, limbHR=6, lumbarE=1, lumbarBT=2)
OUT_OF_PLANE_SUFFIXES = ('_supinate', '_abduct')

# An action interface: 'trq' (torques), 'pos' (target positions), 'poskp' (target positions and
# stiffnesses) or 'flxext' (flexor and extensor activations).
Interface = T.Literal['trq', 'pos', 'poskp', 'flxext']


def make_controller(interface: Interface, segment: rat.RatSegment) -> cx.Transform:
  """Returns a segment's controller of the (non-torque) action interface."""
  size = SEGMENT_SIZES[segment]
  keys = dict(
    obskey_pos=f'proprio_pos_{segment}',
    obskey_vel=f'proprio_vel_{segment}',
    actkey_trq=f'actuators_trq_{segment}',
  )
  if interface == 'flxext':
    return cx.transforms.SpringMuscleController(
      poseq=rat.rat_actuators_pos_mid(segment),
      keq=[0.3] * size,
      trq_max_flx=[0.5] * size,
      trq_max_ext=[0.5] * size,
      actkey_flx=f'muscle_flx_{segment}',
      actkey_ext=f'muscle_ext_{segment}',
      **keys,
    )
  return cx.transforms.ProportionalDerivativeController(
    actionable_pos=f'actuators_pos_{segment}',
    actionable_kp=f'actuators_kp_{segment}' if interface == 'poskp' else None,
    kp=[0.3] * size,
    pos_margin=0.2,
    undamped=True,
    **keys,
  )


def make_rat(
  interface: Interface,
  segments: tuple[rat.RatSegment, ...] = (*LIMB_SEGMENTS, *LUMBAR_SEGMENTS),
  planar: bool = False,
  **kwargs,
) -> rat.Rat:
  """Returns a rat whose `segments` are actionable through the action interface. If `planar`, the
  limbs' out-of-plane DOFs are not actionable."""
  actionable_dofs = {
    segment: tuple(
      name for name in rat.RAT_ACTUATORS[segment]
      if not (planar and name.endswith(OUT_OF_PLANE_SUFFIXES))
    )
    for segment in segments
  }
  controller = {} if interface == 'trq' else {
    segment: make_controller(interface, segment) for segment in segments
  }
  return rat.Rat(
    actionable_dofs=actionable_dofs,
    controller=controller,
    contact_mode='trunk-limbs-tail',
    rescale_mode='limits',
    **kwargs,
  )


class BodyTask(cx.MjxTask):
  """A task of only the rat on a plane, starting from its initial pose."""

  def __init__(self, body: rat.Rat, env_timestep: float = 0.03):
    mj_spec = mj.MjSpec()
    mj_spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    body.mj_spec.default.name = 'body/main'  # As `MoveAtFixedVelocity`, for a loadable spec.
    mj_spec.attach(body.mj_spec, '', frame=mj_spec.worldbody.add_frame())
    super().__init__(
      children=dict(body=body),
      mj_spec=mj_spec,
      env_timestep=env_timestep,
      sim_timestep=0.001,
    )

  def on_reset(self, state, subtask):
    return self.children['body'].init_pose(state)


def make_env(body: rat.Rat, **kwargs) -> tuple[cx.MjxEnv, cx.MjxEnvState]:
  """Returns the native MuJoCo env of a `BodyTask` and its initial state of one world."""
  env = cx.MjxEnv(BodyTask(body, **kwargs), backend='mujoco')
  return env, env.initialize(jax.random.split(jax.random.key(0), 1))


def get_sizes(space: cx.Dict) -> dict[str, int]:
  """Returns the size of each key of a space of one world."""
  return {key: box.shape[-1] for key, box in space.items()}


def get_position_ranges(model: mj.MjModel) -> dict[str, np.ndarray]:
  """Returns each actuator's position range (rad, or m for tendons), by actuator name."""
  ranges = {}
  for actuator_id in range(model.nu):
    target_id = model.actuator_trnid[actuator_id, 0]
    is_joint = model.actuator_trntype[actuator_id] == mj.mjtTrn.mjTRN_JOINT
    ranges[model.actuator(actuator_id).name] = (
      model.jnt_range[target_id] if is_joint else model.tendon_range[target_id]
    )
  return ranges


@pytest.fixture(scope='module')
def model() -> mj.MjModel:
  """The compiled rat model."""
  return mj.MjSpec.from_file(str(rat.RAT_XML)).compile()


# ==================================================================================================
# Tables


def test_actuator_names_match_the_model(model):
  assert model.nu == 38
  assert tuple(model.actuator(i).name for i in range(model.nu)) == rat.RAT_ACTUATOR_NAMES


def test_joint_tables_match_the_model(model):
  joints = [model.joint(i).name for i in range(1, model.njnt)]  # Without the root.

  assert len(rat.RAT_JOINTS_POS_INIT) == len(joints) == 55
  assert rat.POSE_JOINTS == [joint for joint in joints if joint in rat.POSE_JOINTS]
  segment_sizes = {segment: len(names) for segment, names in rat.RAT_ACTUATORS.items()}
  for table in (
    rat.RAT_ACTUATORS_POS_MID,
    rat.RAT_ACTUATORS_POS_MID_01,
    rat.RAT_ACTUATORS_KP,
    rat.RAT_ACTUATORS_KD,
  ):
    assert {segment: len(values) for segment, values in table.items()} == segment_sizes


def test_normalized_midpoints_match_the_widened_position_ranges(model):
  ranges = get_position_ranges(model)

  for segment, names in rat.RAT_ACTUATORS.items():
    low, high = np.array([ranges[name] for name in names]).T
    low, high = low - 0.2 * (high - low), high + 0.2 * (high - low)  # `pos_margin: 0.2`.
    expected = np.clip((rat.RAT_ACTUATORS_POS_MID[segment] - low) / (high - low), 0, 1)
    np.testing.assert_allclose(rat.RAT_ACTUATORS_POS_MID_01[segment], expected, atol=0.01)


# ==================================================================================================
# Rat


@pytest.mark.parametrize(
  'interface, kwargs, prefixes',
  [
    ('trq', dict(actionable_trq=True), ['actuators_trq']),
    ('pos', dict(), ['actuators_pos']),
    ('poskp', dict(), ['actuators_pos', 'actuators_kp']),
    ('flxext', dict(), ['muscle_flx', 'muscle_ext']),
  ],
  ids=['trq', 'pos', 'poskp', 'flxext'],
)
def test_actions_are_the_interface_of_the_limbs_and_lumbar(interface, kwargs, prefixes):
  env, state = make_env(make_rat(interface, **kwargs))

  action_sizes = get_sizes(env.action_space(state).unvectorize())

  assert action_sizes == {
    f'body/{prefix}_{segment}': size
    for prefix in prefixes
    for segment, size in SEGMENT_SIZES.items()
  }


def test_observations_are_the_actionable_proprioception_and_tactile():
  env, state = make_env(make_rat('flxext'))

  observation_sizes = get_sizes(env.observation_space(state).unvectorize())

  assert observation_sizes == {
    **{
      f'body/proprio_{quantity}_{segment}': size
      for quantity in ('pos', 'vel', 'trq')
      for segment, size in SEGMENT_SIZES.items()
    },
    **{f'body/tactile_{segment}': 2 for segment in LIMB_SEGMENTS},
  }


def test_unactionable_lumbar_has_no_lumbar_interface():
  env, state = make_env(make_rat('flxext', LIMB_SEGMENTS))

  keys = [*env.action_space(state).keys(), *env.observation_space(state).keys()]

  assert keys and not any(segment in key for key in keys for segment in LUMBAR_SEGMENTS)


def test_planar_out_of_plane_actuators_ignore_the_muscle_actions():
  # One substep per step, so both steps' torques are computed from the same (reset) state.
  env, state = make_env(make_rat('flxext', planar=True), env_timestep=0.001)
  space = env.action_space(state)
  reset = env.reset(state, jnp.stack([jax.random.key(0)]))

  first, second = (env.step(reset.state, space.sample(jax.random.key(seed))) for seed in (1, 2))

  torques = [np.asarray(step.state.mjx_data.ctrl[0]) for step in (first, second)]
  out_of_plane = [
    rat.RAT_ACTUATOR_NAMES.index(name)
    for name in rat.RAT_ACTUATORS['limbFL'] + rat.RAT_ACTUATORS['limbHL']
    if name.endswith(OUT_OF_PLANE_SUFFIXES)
  ]
  in_plane = [rat.RAT_ACTUATOR_NAMES.index(name) for name in ('elbow_L', 'knee_L')]
  np.testing.assert_array_equal(torques[0][out_of_plane], torques[1][out_of_plane])
  assert not np.array_equal(torques[0][in_plane], torques[1][in_plane])


def test_trunk_limbs_tail_contact_mode_collides_trunk_limbs_and_tail_only():
  body = make_rat('flxext')

  contype = {geom.name: geom.contype for geom in body.mj_spec.geoms}

  assert body.get_termination_contact_geoms() == [
    'torso_collision',
    'pelvis_gut_collision',
    'humerus_L_collision',
    'humerus_R_collision',
    'upper_leg_L_collision',
    'upper_leg_R_collision',
  ]
  assert contype['vertebrae_C16C20_collision'] == contype['vertebrae_C26C30_collision'] == 1
  assert contype['skull_0_collision'] == contype['jaw_F_collision'] == 0


def test_invalid_controllers_are_rejected():
  with pytest.raises(TypeError, match='controller'):
    rat.Rat(controller=42)
