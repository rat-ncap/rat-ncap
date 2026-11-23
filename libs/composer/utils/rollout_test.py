import collections
import gc
import pathlib
import pickle
import types
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
import pytest

import composer.mjx_env as mjx_env
from composer import core
from composer.utils import rollout

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


def make_env() -> mjx_env.MjxEnv:
  """A `BoxTask` env on the 'mujoco' backend, which compiles in seconds on CPU."""
  return mjx_env.MjxEnv(BoxTask(), backend='mujoco')


class CountingPolicy(rollout.Policy[int]):
  """Pushes with zero force; its state counts the steps it has taken."""

  def __init__(self):
    self.steps = 0

  def step(self, observation, rng):
    self.steps += 1
    return {'push': jnp.zeros((rng.shape[0], 1))}

  def get_state(self):
    return self.steps

  def set_state(self, state):
    self.steps = state


# ==================================================================================================
# Rollout


def test_rollout_from_policy_records_reset_and_every_step():
  rollout_data = rollout.rollout_from_policy(
    make_env(),
    duration=0.05,  # 5 steps of 0.01 s.
    record={'steps': lambda ctx: int(ctx.step.state.steps[0])},
  )

  assert rollout_data.record['steps'] == [0, 1, 2, 3, 4, 5]
  assert len(rollout_data.reward) == 6
  assert float(rollout_data.reward[0][0]) == 0.0


def test_rollout_from_policy_records_at_record_rate():
  rollout_data = rollout.rollout_from_policy(
    make_env(),
    duration=0.05,
    record={'steps': lambda ctx: int(ctx.step.state.steps[0])},
    record_rate=50,  # Half the env rate.
  )

  assert rollout_data.record['steps'] == [0, 2, 4]


def test_record_context_has_the_policy_state_that_produced_the_action():
  rollout_data = rollout.rollout_from_policy(
    make_env(),
    policy=CountingPolicy(),
    duration=0.03,
    record={'policy_state': lambda ctx: ctx.policy_state, 'i': lambda ctx: ctx.i},
  )

  assert rollout_data.record['policy_state'] == rollout_data.record['i'] == [0, 1, 2, 3]


@pytest.mark.parametrize('policy', ['zero', 'random', 'cycle'])
def test_rollout_from_policy_builds_policies_from_names(policy):
  env = make_env()

  rollout_data = rollout.rollout_from_policy(
    env,
    policy=policy,
    duration=0.02,
    vectorize=2,
    record={'action': lambda ctx: ctx.action},
  )

  for action in rollout_data.record['action']:
    assert action['push'].shape == (2, 1)
    assert bool(jnp.all(jnp.abs(action['push']) <= 1))


# ==================================================================================================
# Save / Load


def test_load_steps_restores_saved_steps_and_model(tmp_path: pathlib.Path):
  env = make_env()
  steps = rollout.rollout_from_policy(
    env,
    policy='random',
    duration=0.03,
    record={'step': lambda ctx: ctx.step},
  ).record['step']
  filepath = tmp_path / 'steps.pkl'

  rollout.save_steps(filepath, steps)
  loaded = rollout.load_steps(filepath, env)

  assert len(loaded) == len(steps)
  for step, original in zip(loaded, steps):
    np.testing.assert_array_equal(step.state.mjx_data.qpos, original.state.mjx_data.qpos)
    np.testing.assert_array_equal(step.observation['qpos'], original.observation['qpos'])
    np.testing.assert_array_equal(
      step.state.mjx_data._impl.efc_force,
      original.state.mjx_data._impl.efc_force,
    )
    assert step.state.mj_model is not None and step.state.mjx_model is not None



class FakeState(eqx.Module):
  """Minimal stand-in for `MjxEnvState` with the fields `load_steps` reinserts."""
  x: jnp.ndarray
  mj_spec: object = None
  mj_model: object = None
  mjx_model: object = None

  def replace(self, **kwargs) -> 'FakeState':
    return type(self)(**(dict(vars(self)) | kwargs))


def test_load_steps_does_not_retain_unpickled_modules(tmp_path: pathlib.Path):
  # Equinox registers modules created by unpickling in a global set that it never clears, which
  # leaked every step loaded by `load_steps`. Loaded states must be collectable once dropped.
  steps = [
    core.Step(
      state=FakeState(x=jnp.full((4,), float(i))),
      observation={},
      terminated=jnp.zeros((), bool),
      truncated=jnp.zeros((), bool),
      reward=jnp.zeros(()),
      metrics={},
    ) for i in range(3)
  ]
  filepath = tmp_path / 'steps.pkl'
  with open(filepath, 'wb') as f:
    pickle.dump({'static': {}, 'dynamic': steps, 'vectorize': 1}, f)
  spec = mj.MjSpec.from_string('<mujoco><worldbody/></mujoco>')
  env = types.SimpleNamespace(task=types.SimpleNamespace(mj_spec=spec), backend='jax')

  initialising = eqx._module._module._currently_initialising
  size = len(initialising._dict)
  loaded = rollout.load_steps(filepath, env)  # type: ignore

  assert [float(step.state.x[0]) for step in loaded] == [0., 1., 2.]
  assert len(initialising._dict) == size
  # The pickled `dynamic` steps are only referenced through `loaded` (via `dataclasses.replace`
  # inputs, which are dropped), so dropping `loaded` must free everything.
  refs = [weakref.ref(step.state.x) for step in loaded]
  del loaded
  gc.collect()
  assert all(ref() is None for ref in refs)
