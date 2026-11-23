import collections
import gc
import typing as T
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import mujoco.mjx._src.support as mjx_support
import numpy as np
import pytest

import composer.core as core
import composer.mjx_env as mjx_env
import composer.mujoco_physics as mujoco_physics

# ==================================================================================================
# Vectorization


class Scaler(eqx.Module):
  """Equinox module with an array leaf and a static leaf, like the env state's mix of both."""
  weight: jax.Array
  tag: str = eqx.field(static=True)

  def scale(self, x: jax.Array, *, offset: float = 0.) -> 'Scaler':
    """Returns a scaler whose weight is `x * weight + offset`, with the same tag."""
    return Scaler(x * self.weight + offset, self.tag)


def make_jitted_closure() -> tuple[weakref.ref, weakref.ref]:
  """Jits and calls a closure over a fresh module, and returns weak references to both."""
  scaler = Scaler(jnp.full(3, 2.), 'a')
  fn = lambda x: scaler.scale(x)
  mjx_env._filter_jit(fn)(jnp.ones(3))
  return weakref.ref(fn), weakref.ref(scaler.weight)


def test_filter_jit_matches_eqx_filter_jit():
  # A function with array leaves (traced) and str/float/None leaves (static) in its inputs and
  # outputs, including a static module field.
  scaler = Scaler(jnp.arange(3.), 'a')
  fn = lambda x, label, *, offset: (scaler.scale(x, offset=offset), label, None, np.ones(2) * x[0])
  args = (jnp.full(3, 2.), 'label')

  expected = eqx.filter_jit(fn)(*args, offset=1.)
  actual = mjx_env._filter_jit(fn)(*args, offset=1.)

  assert jax.tree.structure(actual) == jax.tree.structure(expected)
  jax.tree.map(np.testing.assert_array_equal, actual, expected)
  assert actual[0].tag == 'a'
  assert actual[1] == 'label'


def test_filter_jit_retraces_on_new_static_values_and_shapes():
  traces: list[tuple[tuple[int, ...], float]] = []

  def fn(x: jax.Array, scale: float) -> jax.Array:
    traces.append((x.shape, scale))
    return x * scale

  jitted = mjx_env._filter_jit(fn)
  np.testing.assert_array_equal(jitted(jnp.ones(2), 2.), [2., 2.])
  np.testing.assert_array_equal(jitted(jnp.zeros(2), 2.), [0., 0.])  # Same signature: cached.
  np.testing.assert_array_equal(jitted(jnp.ones(2), 3.), [3., 3.])  # New static value.
  np.testing.assert_array_equal(jitted(jnp.ones(4), 3.), [3.] * 4)  # New shape.

  assert traces == [((2,), 2.), ((2,), 3.), ((4,), 3.)]


def test_filter_jit_frees_jitted_closures_and_their_captures():
  # `eqx.filter_jit` retains jitted closures and their captures forever (a leak per env or rollout);
  # `_filter_jit` frees them with its wrapper.
  fn_ref, weight_ref = make_jitted_closure()

  gc.collect()

  assert fn_ref() is None
  assert weight_ref() is None


def test_vmap_state_update_applies_out_axes_to_returns():
  env = make_env()
  state = env.initialize(jax.random.split(jax.random.key(0), 2))

  def steps_and_constant(state):
    return state, (state.steps, jnp.ones(3))

  _, (steps, constant) = mjx_env.vmap_state_update(steps_and_constant, out_axes=(0, None))(state)

  assert steps.shape == (2,)
  assert constant.shape == (3,)


# ==================================================================================================
# Environment

# A box resting on a plane, with a motor pushing it along x.
BOX_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <camera name="side" pos="0 -1 0.2" xyaxes="1 0 0 0 0 1"/>
    <body name="box" pos="0 0 0.05">
      <freejoint name="slide"/>
      <geom name="box" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="push" joint="slide" gear="1 0 0 0 0 0"/></actuator>
</mujoco>
"""


class BoxTask(mjx_env.MjxTask):
  """Pushes the box with action `push`; observes its `qpos`. Rewards 1 per step, and terminates
  from step `terminate_at`. Each episode starts the box at a random x in [0, 0.1]."""
  terminate_at: int | None

  def __init__(self, terminate_at: int | None = None):
    super().__init__(
      mj_spec=mj.MjSpec.from_string(BOX_XML),
      env_timestep=0.01,
      sim_timestep=0.002,
    )
    self.terminate_at = terminate_at

  def action_space(self, state):
    return core.Dict({'push': core.Box(-1.0, 1.0, shape=(1,))})

  def action(self, state, action):
    return state.replace(mjx_data=state.mjx_data.replace(ctrl=action['push']))

  def observation_space(self, state):
    return core.Dict({'qpos': core.Box(-jnp.inf, jnp.inf, shape=(7,))})

  def observation(self, state):
    return collections.OrderedDict(qpos=state.mjx_data.qpos), state

  def on_reset(self, state, subtask):
    x = jax.random.uniform(state.rng, (), maxval=0.1)
    return state.replace(mjx_data=state.mjx_data.replace(qpos=state.mjx_data.qpos.at[0].set(x)))

  def get_terminated(self, state):
    if self.terminate_at is None:
      return jnp.zeros((), dtype=bool)
    return state.steps >= self.terminate_at

  def get_rewards(self, state, terminated):
    return {'alive': (jnp.ones(()), jnp.ones(()))}


def make_env(terminate_at: int | None = None, truncate: int | None = None) -> mjx_env.MjxEnv:
  """A `BoxTask` env on the 'mujoco' backend, which compiles in seconds on CPU."""
  return mjx_env.MjxEnv(BoxTask(terminate_at), truncate=truncate, backend='mujoco')


def reset_worlds(env: mjx_env.MjxEnv, seeds: list[int]) -> core.Step:
  """Initializes and resets one world per seed."""
  rng = jnp.stack([jax.random.key(seed) for seed in seeds])
  return env.reset(env.initialize(rng), rng)


def push(value: float, size: int) -> dict[str, jax.Array]:
  """The action pushing every world's box by `value`."""
  return {'push': jnp.full((size, 1), value)}


def test_reset_gives_same_observation_for_same_key():
  env = make_env()

  first = reset_worlds(env, [0, 1])
  second = reset_worlds(env, [0, 1])

  np.testing.assert_array_equal(first.observation['qpos'], second.observation['qpos'])
  assert not np.array_equal(first.observation['qpos'][0], first.observation['qpos'][1])


def test_step_of_two_worlds_matches_each_world_alone():
  env = make_env()
  both = env.step(reset_worlds(env, [0, 1]).state, {'push': jnp.array([[1.0], [-1.0]])})

  first = env.step(reset_worlds(env, [0]).state, push(1.0, 1))
  second = env.step(reset_worlds(env, [1]).state, push(-1.0, 1))

  np.testing.assert_allclose(both.state.mjx_data.qpos[0], first.state.mjx_data.qpos[0], atol=1e-6)
  np.testing.assert_allclose(both.state.mjx_data.qpos[1], second.state.mjx_data.qpos[0], atol=1e-6)


def test_masked_reset_resets_only_masked_worlds():
  env = make_env()
  step = env.step(reset_worlds(env, [0, 1]).state, push(1.0, 2))

  mask = jnp.array([True, False])

  reset = env.reset(step.state, jax.random.split(jax.random.key(2), 2), reset=(mask, step))

  assert reset.state.steps.tolist() == [0, 1]
  np.testing.assert_array_equal(reset.observation['qpos'][1], step.observation['qpos'][1])
  np.testing.assert_array_equal(reset.state.mjx_data.qpos[1], step.state.mjx_data.qpos[1])
  assert not np.array_equal(reset.observation['qpos'][0], step.observation['qpos'][0])


def test_terminated_is_sticky_and_zeroes_later_rewards():
  env = make_env(terminate_at=2)
  state = reset_worlds(env, [0]).state

  steps = []
  for _ in range(4):
    step = env.step(state, push(0.0, 1))
    steps.append(step)
    state = step.state

  assert [bool(step.terminated[0]) for step in steps] == [False, True, True, True]
  assert [float(step.reward[0]) for step in steps] == [1.0, 1.0, 0.0, 0.0]


def test_truncated_from_truncate_step_on():
  env = make_env(truncate=2)
  state = reset_worlds(env, [0]).state

  first = env.step(state, push(0.0, 1))
  second = env.step(first.state, push(0.0, 1))
  third = env.step(second.state, push(0.0, 1))

  assert [bool(step.truncated[0]) for step in (first, second, third)] == [False, True, True]


def test_spaces_are_vectorized_and_sampled_actions_validate():
  env = make_env()
  state = reset_worlds(env, [0, 1, 2]).state

  action_space = env.action_space(state)
  observation_space = env.observation_space(state)

  assert action_space['push'].shape == (3, 1)
  assert observation_space['qpos'].shape == (3, 7)
  action_space.validate(action_space.sample(jax.random.key(0)))


# ==================================================================================================
# Contact forces


def test_contact_force_on_resting_box_balances_its_weight():
  env = make_env()
  state = reset_worlds(env, [0]).state
  for _ in range(50):  # Let the box settle (0.5 s).
    state = env.step(state, push(0.0, 1)).state
  box = state.mj_model.body('box').id

  force = mjx_env.vmap_state(
    lambda state: mjx_env.contact_force_body_world(state.mjx_model, state.mjx_data, [box])
  )(state)  # [world, body, 3]

  weight = float(state.mj_model.body_mass[box] * -state.mj_model.opt.gravity[2])
  np.testing.assert_allclose(force[0, 0], [0.0, 0.0, weight], atol=0.05 * weight)


# Contacts of condim 1, 3 and 4 between three spheres and a plane.
CONTACTS_XML = """
<mujoco>
  <option cone="{cone}"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" condim="3"/>
    <body pos="0 0 0.09"><freejoint/><geom type="sphere" size="0.1" condim="1"/></body>
    <body pos="0.3 0 0.09"><freejoint/><geom type="sphere" size="0.1" condim="3"/></body>
    <body pos="0.6 0 0.09"><freejoint/><geom type="sphere" size="0.1" condim="4"/></body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize('cone', ['pyramidal', 'elliptic'])
def test_contact_forces_match_mjx_contact_force_per_contact(cone):
  mj_model = mj.MjModel.from_xml_string(CONTACTS_XML.format(cone=cone))
  mjx_model = mjx.put_model(mj_model, impl='jax')
  data = mujoco_physics.MujocoPhysics(mj_model).forward(mjx.make_data(mj_model, impl='jax'))
  contact = T.cast(mjx.DataJAX, data._impl).contact
  ncon = contact.dim.shape[0]
  expected = jnp.stack([
    mjx_support.contact_force(mjx_model, data, i, to_world_frame=True)[:3] for i in range(ncon)
  ])

  actual, geoms = mjx_env._contact_force_world_jax(mjx_model, data)

  assert float(jnp.abs(expected).max()) > 0  # The spheres press on the plane.
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
  np.testing.assert_array_equal(geoms, contact.geom)


# ==================================================================================================
# Rendering


def has_opengl() -> bool:
  """Whether an OpenGL context can be created (set `MUJOCO_GL`, e.g. to 'egl' or 'osmesa')."""
  try:
    mj.GLContext(8, 8).free()
    return True
  except Exception:
    return False


@pytest.mark.skipif(not has_opengl(), reason='No OpenGL context.')
@pytest.mark.parametrize('depth', [False, True], ids=['rgb', 'depth'])
def test_render_mujoco_gives_one_image_per_state_camera_and_world(depth):
  env = make_env()
  states = [reset_worlds(env, [0, 1, 2]).state] * 2

  images = mjx_env.render(
    states,
    index=[0, 2],
    camera=['side'],
    size=(24, 32),
    depth=depth,
  )

  assert len(images) == 2
  assert images[0].shape == ((1, 2, 24, 32) if depth else (1, 2, 24, 32, 3))


def test_scene_option_survives_serialization_for_workers():
  scene_option = mjx_env._make_scene_option(
    geomgroup=(0, 2),
    sitegroup=(1,),
    jointgroup=(),
    tendongroup=(0,),
    actuatorgroup=(0,),
    flexgroup=(0,),
    skingroup=(0,),
    label='geom',
    frame='world',
    vis={'contactforce': True},
  )

  restored = mjx_env.deserialize_scene_option(mjx_env.serialize_scene_option(scene_option))

  np.testing.assert_array_equal(restored.geomgroup, [1, 0, 1, 0, 0, 0])
  np.testing.assert_array_equal(restored.flags, scene_option.flags)
  assert restored.label == mj.mjtLabel.mjLABEL_GEOM
  assert restored.frame == mj.mjtFrame.mjFRAME_WORLD
