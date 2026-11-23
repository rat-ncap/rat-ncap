import jax
import jax.numpy as jnp
import numpy as np
import pytest

import composer as cx
from envs import arenas, bodies
from envs.tasks import move_at_fixed_velocity as task_module

# A subtask of moving at 0.4 m/s, rewarded for the velocity (not the pose).
SUBTASK = dict(
  target_vx=0.4,
  reward_passive=0.2,
  reward_vx_sparse=0.2,
  reward_vx_dense=0.6,
  reward_pose=0.,
)
MARGIN_VX = 0.1  # The task's default `margin_vx`.
ENV_TIMESTEP = 0.002  # 2 substeps per step.


def make_task(
  arena: cx.MjxEntity | None = None,
  **kwargs,
) -> task_module.MoveAtFixedVelocity_TrackSmooth:
  """The task of a position-controlled rat whose trunk, limbs and tail collide."""
  return task_module.MoveAtFixedVelocity_TrackSmooth(
    bodies.Rat(contact_mode='trunk-limbs-tail'),
    arena=arena,
    env_timestep=ENV_TIMESTEP,
    sim_timestep=0.001,
    **kwargs,
  )


def make_env(task: task_module.MoveAtFixedVelocity_TrackSmooth) -> cx.MjxEnv:
  """A native MuJoCo env of the task."""
  return cx.MjxEnv(task, backend='mujoco')


def reset(env: cx.MjxEnv, worlds: int = 1, **subtask: float) -> cx.Step[cx.MjxEnvState]:
  """Resets `worlds` worlds with `SUBTASK`, updated by `subtask`."""
  rng = jax.random.split(jax.random.key(0), worlds)
  subtask = SUBTASK | subtask
  return env.reset(
    env.initialize(rng),
    rng,
    subtask={key: jnp.full((worlds,), value, dtype=jnp.float32) for key, value in subtask.items()},
  )


def forward(env: cx.MjxEnv, state: cx.MjxEnvState, **mjx_data: jax.Array) -> cx.MjxEnvState:
  """Returns the state with the given `mjx_data` fields, and its derived fields recomputed."""
  state = state.replace(mjx_data=state.mjx_data.replace(**mjx_data))
  state, _ = cx.vmap_state_update(
    lambda world: (world.replace(mjx_data=env.physics.forward(world)), None),
    mj_model_vecfields=env.task.mj_model_vecfields,
  )(state)
  return state


def get_reward_values(env: cx.MjxEnv, state: cx.MjxEnvState) -> dict[str, np.ndarray]:
  """Returns each reward term's value in each world."""
  values = cx.vmap_state(
    lambda world: {
      name: value for name, (_, value) in env.task.get_rewards(world, jnp.array(False)).items()
    },
    mj_model_vecfields=env.task.mj_model_vecfields,
  )(state)
  return {name: np.asarray(value) for name, value in values.items()}


@pytest.fixture(scope='module')
def env() -> cx.MjxEnv:
  """The env of the task on a flat floor."""
  return make_env(make_task())


# ==================================================================================================
# MoveAtFixedVelocity


@pytest.mark.parametrize(
  'root_height, root_rpy, terminated',
  [
    (None, None, False),
    (0.5, (80., 0., 0.), True),
    (0.5, (0., 80., 0.), True),
    (0., None, True),
  ],
  ids=['initial_pose', 'rolled', 'pitched', 'torso_on_the_floor'],
)
def test_terminates_on_head_roll_pitch_or_trunk_contact(env, root_height, root_rpy, terminated):
  state = reset(env).state
  qpos = state.mjx_data.qpos
  if root_height is not None:
    qpos = qpos.at[:, 2].set(root_height)
  if root_rpy is not None:
    qpos = qpos.at[:, 3:7].set(cx.math.rpy_to_quat(jnp.deg2rad(jnp.array(root_rpy))))
  state = forward(env, state, qpos=qpos)

  is_terminated = cx.vmap_state(env.task.get_terminated)(state)

  assert bool(is_terminated[0]) == terminated


def test_contact_sensors_refer_to_the_bumps_terrain():
  np.random.seed(0)
  task = make_task(arenas.Bumps(size=(1, 1), bump_scale=0.05, resolution=50))

  # MJX's 'jax' implementation has no hfield-ellipsoid collisions, so compile with MuJoCo.
  model = task.mj_spec.compile()

  floor_id = model.geom('arena/floor').id
  sensors = [model.sensor(sensor.name) for sensor in task._contact_sensors]
  assert sensors and all(sensor.refid[0] == floor_id for sensor in sensors)


# ==================================================================================================
# MoveAtFixedVelocity_TrackSmooth


@pytest.mark.parametrize(
  'term, margin',
  [('reward_vx_sparse', MARGIN_VX), ('reward_vx_dense', SUBTASK['target_vx'])],
  ids=['sparse', 'dense'],
)
def test_velocity_rewards_peak_at_the_target_and_fall_to_zero_at_the_margin(env, term, margin):
  offsets = margin * np.array([-1.5, -1., -0.5, -0.25, 0., 0.25, 0.5, 1., 1.5])
  state = reset(env, worlds=len(offsets)).state
  qvel = state.mjx_data.qvel.at[:, 0].set(SUBTASK['target_vx'] + offsets)  # Root global vx.

  values = get_reward_values(env, forward(env, state, qvel=qvel))[term]

  within = np.abs(offsets) <= margin
  rising, falling = within & (offsets <= 0), within & (offsets >= 0)
  np.testing.assert_allclose(values[offsets == 0], 1., atol=1e-5)
  np.testing.assert_allclose(values[np.abs(offsets) >= margin], 0., atol=1e-5)
  assert (np.diff(values[rising]) > 0).all() and (np.diff(values[falling]) < 0).all()


def test_reward_is_the_sum_of_the_terms_weighted_by_the_subtask(env):
  state = reset(env).state
  action = env.action_space(state).sample(jax.random.key(1))

  step = env.step(state, action)

  metrics = {name: float(value[0]) for name, value in step.metrics.items()}
  terms = [name.removeprefix('reward/value/') for name in metrics if 'reward/value/' in name]
  weights = {term: metrics[f'reward/weight/{term}'] for term in terms}
  values = {term: metrics[f'reward/value/{term}'] for term in terms}
  subtask_weights = {term: weight for term, weight in SUBTASK.items() if term != 'target_vx'}
  assert {term: weights[term] for term in subtask_weights} == pytest.approx(subtask_weights)
  np.testing.assert_allclose(
    step.reward[0],
    sum(weights[term] * values[term] for term in terms),
    rtol=1e-6,
  )


@pytest.mark.parametrize('stand_time', [0., 2 * ENV_TIMESTEP], ids=['none', 'two_steps'])
def test_target_is_zero_and_pose_rewarded_until_the_stand_time(stand_time):
  env = make_env(make_task(stand_time=stand_time))
  step = reset(env)
  space = env.action_space(step.state)
  targets, pose_values = [float(step.observation['target_vx'][0, 0])], []

  for seed in range(4):
    step = env.step(step.state, space.sample(jax.random.key(seed)))
    targets.append(float(step.observation['target_vx'][0, 0]))
    pose_values.append(float(step.metrics['reward/value/reward_pose'][0]))

  standing = [steps * ENV_TIMESTEP < stand_time for steps in range(5)]  # From the reset on.
  expected = [0. if is_standing else SUBTASK['target_vx'] for is_standing in standing]
  assert targets == pytest.approx(expected)
  assert [value < 1 for value in pose_values] == standing[1:]


def test_pose_reward_tracks_the_initial_pose_while_standing_only(env):
  standing, moving = reset(env, target_vx=0.).state, reset(env).state
  qpos = standing.mjx_data.qpos.at[:, 7:].add(0.2)  # Every joint off its initial position.

  standing_value, moving_value = (
    get_reward_values(env, forward(env, state, qpos=qpos))['reward_pose'][0]
    for state in (standing, moving)
  )

  assert standing_value < 1 and moving_value == 1


def test_reset_is_deterministic(env):
  first, second = reset(env), reset(env)

  for a, b in zip(
    jax.tree.leaves((first.observation, first.state.mjx_data.qpos)),
    jax.tree.leaves((second.observation, second.state.mjx_data.qpos)),
    strict=True,
  ):
    np.testing.assert_array_equal(a, b)
