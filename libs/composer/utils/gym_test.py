import collections
import typing as T

import gymnasium.spaces
import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
import pytest
import torch

import composer.core as core
import composer.mjx_env as mjx_env
import composer.utils.gym as cxg

# ==================================================================================================
# Conversion


def test_to_gym_space_keeps_bounds_shape_dtype_metadata_and_key_order():
  space = core.Dict({
    'b': core.Box(-1.0, 1.0, shape=(2,), labels=['joint'], name='b_name', desc='b_desc'),
    'a': core.Box(jnp.array([0, 1]), jnp.array([5, 6]), dtype=jnp.int32),
  })

  gym_space = cxg.to_gym_space(space)

  assert isinstance(gym_space, gymnasium.spaces.Dict)
  assert list(gym_space.spaces) == ['b', 'a']
  box = gym_space['b']
  assert isinstance(box, cxg.Box)
  assert box.shape == (2,) and box.dtype == np.float32
  np.testing.assert_array_equal(box.low, [-1.0, -1.0])
  assert (box.name, box.desc, box.labels) == ('b_name', 'b_desc', ('joint',))
  int_box = T.cast(cxg.Box, gym_space['a'])
  assert int_box.dtype == np.int32
  np.testing.assert_array_equal(int_box.high, [5, 6])


def test_to_gym_space_rejects_unknown_space():
  with pytest.raises(NotImplementedError):
    cxg.to_gym_space(object())


# ==================================================================================================
# Flattening


def make_nested_space() -> gymnasium.spaces.Dict:
  """A dict of a [2, 3] box and a nested dict of a [4] box."""
  return gymnasium.spaces.Dict(collections.OrderedDict(
    matrix=gymnasium.spaces.Box(-1.0, 1.0, shape=(2, 3)),
    nested=gymnasium.spaces.Dict(collections.OrderedDict(
      vector=gymnasium.spaces.Box(-1.0, 1.0, shape=(4,)),
    )),
  ))


def test_pack_tensors_flattens_with_batch_axes():
  value = {'matrix': torch.ones(5, 7, 2, 3), 'nested': {'vector': torch.zeros(5, 7, 4)}}

  packed = cxg.pack_tensors(make_nested_space(), value)

  assert packed.shape == (5, 7, 10)
  torch.testing.assert_close(packed[..., :6], torch.ones(5, 7, 6))


def test_unpack_tensors_inverts_pack_tensors_with_batch_axes():
  space = make_nested_space()
  value = {
    'matrix': torch.arange(5 * 7 * 6, dtype=torch.float32).reshape(5, 7, 2, 3),
    'nested': {'vector': -torch.arange(5 * 7 * 4, dtype=torch.float32).reshape(5, 7, 4)},
  }

  unpacked = T.cast(dict, cxg.unpack_tensors(space, cxg.pack_tensors(space, value)))

  torch.testing.assert_close(unpacked['matrix'], value['matrix'])
  torch.testing.assert_close(unpacked['nested']['vector'], value['nested']['vector'])


# ==================================================================================================
# Gymnasium adapter

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


def make_gym_env(worlds: int, terminate_at: int | None = None) -> cxg.ComposerEnv:
  """An initialized `ComposerEnv` of a `BoxTask` on the 'mujoco' backend (compiles in seconds on
  CPU)."""
  env = cxg.ComposerEnv(mjx_env.MjxEnv(BoxTask(terminate_at), backend='mujoco'))
  env.initialize(torch.arange(worlds))
  return env


def test_initialize_sets_spaces_of_one_world():
  env = make_gym_env(worlds=3)

  observation_space = T.cast(gymnasium.spaces.Dict, env.observation_space)
  action_space = T.cast(gymnasium.spaces.Dict, env.action_space)
  assert observation_space['qpos'].shape == (7,)
  assert action_space['push'].shape == (1,)


def test_reset_gives_same_observations_for_same_seed():
  env = make_gym_env(worlds=2)

  first = T.cast(dict, env.reset(seed=3)[0])
  second = T.cast(dict, env.reset(seed=3)[0])
  unseeded = T.cast(dict, env.reset()[0])

  torch.testing.assert_close(first['qpos'], second['qpos'])
  assert unseeded['qpos'].shape == (2, 7)


def test_step_returns_final_observations_and_autoreset_new_observations():
  env = make_gym_env(worlds=2, terminate_at=2)
  env.reset(seed=0)
  actions = {'push': torch.zeros(2, 1)}

  first = env.step(actions)
  second = env.step(actions)

  observations, rewards, terminations, truncations, info = first
  observations = T.cast(dict, observations)
  assert not terminations.any()
  torch.testing.assert_close(info['new_observations']['qpos'], observations['qpos'])

  observations, rewards, terminations, truncations, info = second
  observations = T.cast(dict, observations)
  assert terminations.all()
  assert env.state.steps.tolist() == [0, 0]  # Reset after the episode ended.
  assert not torch.equal(info['new_observations']['qpos'], observations['qpos'])
  assert 'metrics/reward/value/alive' in info
