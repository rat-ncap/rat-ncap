import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import pytest

from composer import mujoco_physics

# A free body with a hinged capsule and a sliding box, a tendon, actuators and sensors, resting on a
# plane. It makes contacts of condim 1, 3 and 4 (pyramidal), joint limits and joint friction, so
# every conversion rule is exercised.
XML = """
<mujoco>
  <option timestep="0.002" jacobian="{jacobian}"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" condim="3"/>
    <body name="torso" pos="0 0 0.12">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" condim="4"/>
      <site name="imu"/>
      <body name="leg" pos="0.1 0 0">
        <joint name="hip" type="hinge" axis="0 1 0" range="-60 60" limited="true"
               frictionloss="0.01"/>
        <geom name="shin" type="capsule" fromto="0 0 0 0.15 0 -0.1" size="0.03" condim="1"/>
        <site name="foot" pos="0.15 0 -0.1"/>
      </body>
      <body name="box" pos="-0.12 0 -0.03">
        <joint name="slide" type="slide" axis="1 0 0" range="-0.02 0.02" limited="true"/>
        <geom type="box" size="0.03 0.03 0.03"/>
      </body>
    </body>
  </worldbody>
  <tendon><spatial name="tendon"><site site="imu"/><site site="foot"/></spatial></tendon>
  <actuator><motor joint="hip" gear="1"/><motor tendon="tendon" gear="1"/></actuator>
  <sensor><accelerometer site="imu"/><touch site="foot"/><jointpos joint="hip"/></sensor>
</mujoco>
"""

State = dict[str, np.ndarray]


def make_model(jacobian: str = 'dense') -> mujoco.MjModel:
  """The test model, with a 'dense' or 'sparse' constraint Jacobian."""
  return mujoco.MjModel.from_xml_string(XML.format(jacobian=jacobian))


def make_state(mj_model: mujoco.MjModel, seed: int) -> State:
  """A random `qpos`, `qvel` and `ctrl`: low enough for contacts, sometimes past a joint limit."""
  rng = np.random.default_rng(seed)
  mj_data = mujoco.MjData(mj_model)

  # Torso height and a small tilt.
  mj_data.qpos[2] = 0.11 + 0.02 * rng.random()
  quat = np.array([1, 0.05 * rng.standard_normal(), 0.05 * rng.standard_normal(), 0])
  mj_data.qpos[3:7] = quat / np.linalg.norm(quat)

  # Hip angle, and the slide sometimes past its limit.
  mj_data.qpos[7] = rng.uniform(-1.2, 1.2)
  mj_data.qpos[8] = rng.uniform(-0.03, 0.03)

  mj_data.qvel[:] = 0.3 * rng.standard_normal(mj_model.nv)
  mj_data.ctrl[:] = rng.uniform(-1, 1, mj_model.nu)
  return dict(qpos=mj_data.qpos.copy(), qvel=mj_data.qvel.copy(), ctrl=mj_data.ctrl.copy())


def make_mjx_batch(mj_model: mujoco.MjModel, states: list[State]) -> mjx.Data:
  """A batch of 'jax'-layout data holding `states`, with stale derived fields."""
  data = mjx.make_data(mj_model, impl='jax')
  batch = [
    data.replace(**{name: jnp.asarray(value, data.qpos.dtype) for name, value in state.items()})
    for state in states
  ]
  return jax.tree.map(lambda *leaves: jnp.stack(leaves), *batch)


def run_mujoco(mj_model: mujoco.MjModel, state: State, run) -> mujoco.MjData:
  """Runs C MuJoCo's `run` (`mj_step` or `mj_forward`) from `state` rounded to float32, as the
  physics receives it."""
  mj_data = mujoco.MjData(mj_model)
  for name, value in state.items():
    getattr(mj_data, name)[...] = np.asarray(value, np.float32)
  run(mj_model, mj_data)
  return mj_data


def assert_trees_close(actual: mjx.Data, expected: mjx.Data):
  """Asserts both trees have the same leaves, equal to float32 precision, naming any mismatch."""
  actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
  expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
  assert [key for key, _ in actual_leaves] == [key for key, _ in expected_leaves]
  for (key, a), (_, e) in zip(actual_leaves, expected_leaves):
    np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6, err_msg=jax.tree_util.keystr(key))


# ==================================================================================================
# Physics


@pytest.mark.parametrize('jacobian', ['dense', 'sparse'])
def test_forward_matches_mjx_put_data_for_every_leaf(jacobian):
  mj_model = make_model(jacobian)
  states = [make_state(mj_model, seed) for seed in range(3)]
  references = [run_mujoco(mj_model, state, mujoco.mj_forward) for state in states]
  expected = jax.tree.map(
    lambda *leaves: jnp.stack(leaves),
    *[mjx.put_data(mj_model, reference, impl='jax') for reference in references],
  )

  physics = mujoco_physics.MujocoPhysics(mj_model)
  actual = jax.jit(jax.vmap(physics.forward))(make_mjx_batch(mj_model, states))

  assert max(reference.ncon for reference in references) > 0  # Contacts were converted.
  assert_trees_close(actual, expected)


def test_step_matches_mj_step():
  mj_model = make_model()
  states = [make_state(mj_model, seed) for seed in range(4)]
  references = [run_mujoco(mj_model, state, mujoco.mj_step) for state in states]

  physics = mujoco_physics.MujocoPhysics(mj_model)
  actual = jax.jit(jax.vmap(physics.step))(make_mjx_batch(mj_model, states))

  np.testing.assert_allclose(
    actual.qpos,
    np.stack([reference.qpos for reference in references]),
    rtol=1e-6,
    atol=1e-7,
  )
  np.testing.assert_allclose(
    actual.qvel,
    np.stack([reference.qvel for reference in references]),
    rtol=1e-5,
    atol=1e-6,
  )


def test_step_gives_same_result_when_repeated_with_reused_mj_datas():
  mj_model = make_model()
  data = make_mjx_batch(mj_model, [make_state(mj_model, seed) for seed in range(4)])
  physics = mujoco_physics.MujocoPhysics(mj_model)
  step = jax.jit(jax.vmap(physics.step))

  first = step(data)
  second = step(data)

  assert len(physics.mj_datas) == 4
  assert_trees_close(second, first)


def test_nested_vmap_matches_flat_vmap():
  mj_model = make_model()
  data = make_mjx_batch(mj_model, [make_state(mj_model, seed) for seed in range(6)])
  nested_data = jax.tree.map(lambda x: x.reshape((2, 3) + x.shape[1:]), data)
  physics = mujoco_physics.MujocoPhysics(mj_model)

  nested = jax.jit(jax.vmap(jax.vmap(physics.forward)))(nested_data)
  flat = jax.jit(jax.vmap(physics.forward))(data)

  assert_trees_close(jax.tree.map(lambda x: x.reshape((6,) + x.shape[2:]), nested), flat)


def test_get_physics_returns_one_physics_per_model():
  mj_model = make_model()
  other_model = make_model()

  physics = mujoco_physics.get_physics(mj_model)

  assert mujoco_physics.get_physics(mj_model) is physics
  assert mujoco_physics.get_physics(other_model) is not physics

