import collections

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from composer import core, transforms

# ==================================================================================================
# Helpers


class EnvState(core.BaseEnvState):
  """A minimal env state with the substep fields that some transforms read."""
  sim_timestep: float = eqx.field(static=True)
  sim_substeps: int = eqx.field(static=True)
  substeps: jax.Array


def box(low, high, shape=(2,)) -> core.Box:
  """A `Box` with the given bounds."""
  return core.Box(low=low, high=high, shape=shape)


def initialize[Transform: core.Transform](
  transform: Transform,
  observation_space: core.Dict,
  action_space: core.Dict,
  sim_substeps: int = 1,
) -> tuple[Transform, EnvState]:
  """Returns the transform with its state index set, and an initialized state."""
  transform = core.initialize_state_index(transform)
  state = EnvState(
    rng=jax.random.key(0),
    steps=jnp.array(0),
    terminated=jnp.array(False),
    local=core.State(transform, frozen=False),
    sim_timestep=0.001,
    sim_substeps=sim_substeps,
    substeps=jnp.array(0),
  )
  state = transform.initialize(state, {}, observation_space, action_space)
  return transform, state


def observe(transform, observation_space, state, pos, vel) -> EnvState:
  """Passes joint positions and velocities through the transform's observation."""
  value = collections.OrderedDict(joints_pos=jnp.asarray(pos), joints_vel=jnp.asarray(vel))
  _, state = transform.observation(observation_space, value, state)
  return state


JOINT_SPACES = core.Dict({'joints_pos': box(-1.0, 1.0), 'joints_vel': box(-10.0, 10.0)})
TORQUE_SPACE = core.Dict({'joints_trq': box(-10.0, 10.0)})

# ==================================================================================================
# Clipping


def test_clip_action_clips_only_listed_keys():
  space = core.Dict({'a': box(-1.0, 1.0), 'b': box(-1.0, 1.0)})
  transform = transforms.ClipAction(keys=['a'])
  value = collections.OrderedDict(a=jnp.array([-5.0, 5.0]), b=jnp.array([-5.0, 5.0]))

  action, _ = transform.action(space, value, None)

  np.testing.assert_array_equal(action['a'], [-1.0, 1.0])
  np.testing.assert_array_equal(action['b'], [-5.0, 5.0])


# ==================================================================================================
# Buffers


@pytest.mark.parametrize(
  'sim_substeps, delay_substeps',
  [(10, 0), (10, 5), (10, 10), (10, 15), (1, 0), (1, 2)],
)
def test_buffer_action_substeps_delays_actions_by_the_configured_substeps(
  sim_substeps,
  delay_substeps,
):
  space = core.Dict({'a': box(-10.0, 10.0, shape=(1,))})
  transform, state = initialize(
    transforms.BufferActionSubsteps(delay=delay_substeps * 0.001),
    core.Dict(),
    space,
    sim_substeps=sim_substeps,
  )

  # Env step `i` (from 1) takes action `i`; record the action applied at each substep.
  applied = []
  for step in range(1, 5):
    for _ in range(sim_substeps):
      state = state.replace(substeps=state.substeps + 1)
      action, state = transform.action(space, {'a': jnp.array([float(step)])}, state)
      applied.append(float(action['a'][0]))

  # The action applied at substep `n` (from 0) is the one current `delay` substeps earlier (zero
  # before the first).
  expected = [
    0.0 if n < delay_substeps else float((n - delay_substeps) // sim_substeps + 1)
    for n in range(4 * sim_substeps)
  ]
  assert applied == expected


# ==================================================================================================
# Rescaling


def test_rescale_action_maps_scale_to_bounds_and_back():
  space = core.Dict({'a': box(2.0, 6.0)})
  transform = transforms.RescaleAction(signed=False)

  rescaled_space = transform.action_space(space, None)
  action, _ = transform.action(space, {'a': jnp.array([-1.0, 1.0])}, None)
  middle, _ = transform.action(space, {'a': jnp.array([0.0, 0.0])}, None)

  np.testing.assert_allclose(rescaled_space['a'].low, [-1.0, -1.0])
  np.testing.assert_allclose(rescaled_space['a'].high, [1.0, 1.0])
  np.testing.assert_allclose(action['a'], [2.0, 6.0])
  np.testing.assert_allclose(middle['a'], [4.0, 4.0])


def test_signed_rescale_action_of_a_range_excluding_zero_uses_the_upper_half_of_scale():
  space = core.Dict({'a': box(2.0, 6.0)})
  transform = transforms.RescaleAction(signed=True)

  rescaled_space = transform.action_space(space, None)
  action, _ = transform.action(space, {'a': jnp.array([0.0, 1.0])}, None)

  np.testing.assert_allclose(rescaled_space['a'].low, [0.0, 0.0])
  np.testing.assert_allclose(rescaled_space['a'].high, [1.0, 1.0])
  np.testing.assert_allclose(action['a'], [2.0, 6.0])


def test_rescale_accepts_zero_scalar_bounds():
  space = core.Dict({'a': box(-5.0, 5.0)})
  transform = transforms.RescaleObservation(min=0.0, max=1.0, signed=False)

  observation, _ = transform.observation(space, {'a': jnp.array([0.0, 1.0])}, None)

  np.testing.assert_allclose(observation['a'], [-1.0, 1.0])


@pytest.mark.parametrize(
  'transform_type',
  [transforms.RescaleObservation, transforms.RescaleAction],
)
def test_rescale_rejects_only_one_bound(transform_type):
  with pytest.raises(AssertionError, match='both min and max'):
    transform_type(min=0.0)


def test_rescale_observation_maps_bounds_to_scale_and_clips():
  space = core.Dict({'a': box(2.0, 6.0), 'kept': box(2.0, 6.0)})
  transform = transforms.RescaleObservation(signed=False, exclude=['kept'])
  value = collections.OrderedDict(a=jnp.array([2.0, 10.0]), kept=jnp.array([2.0, 10.0]))

  observation, _ = transform.observation(space, value, None)

  np.testing.assert_allclose(observation['a'], [-1.0, 1.0])
  np.testing.assert_allclose(observation['kept'], [2.0, 10.0])


# ==================================================================================================
# Reducing, overriding and mapping


def test_reduce_action_inserts_constants_and_removes_fully_constrained_keys():
  space = core.Dict({'a': box(-1.0, 1.0, shape=(3,)), 'b': box(-1.0, 1.0)})
  transform = transforms.ReduceAction(constrain={'a': [None, 0.5, None], 'b': [0.1, 0.2]})

  reduced_space = transform.action_space(space, None)
  action, _ = transform.action(space, {'a': jnp.array([7.0, 9.0])}, None)

  assert list(reduced_space.keys()) == ['a']
  assert reduced_space['a'].shape == (2,)
  np.testing.assert_allclose(action['a'], [7.0, 0.5, 9.0])
  np.testing.assert_allclose(action['b'], [0.1, 0.2])


def test_override_action_overwrites_only_listed_indices():
  space = core.Dict({'a': box(-1.0, 1.0, shape=(3,))})
  transform = transforms.OverrideAction(override={'a': [None, 0.5, None]})

  action, _ = transform.action(space, {'a': jnp.array([7.0, 8.0, 9.0])}, None)

  np.testing.assert_allclose(action['a'], [7.0, 0.5, 9.0])
  assert transform.action_space(space, None) is space


def test_map_action_splits_a_key_and_gathers_it_back():
  space = core.Dict({
    'a': core.Box(-1.0, 1.0, shape=(3,), labels=[('x', 'y', 'z')]),
    'other': box(-1.0, 1.0),
  })
  transform = transforms.MapAction({
    'a_first': ('a', [True, False, True]),
    'a_second': ('a', [False, True, False]),
  })
  value = collections.OrderedDict(
    a_first=jnp.array([1.0, 3.0]),
    a_second=jnp.array([2.0]),
    other=jnp.array([4.0, 5.0]),
  )

  mapped_space = transform.action_space(space, None)
  action, _ = transform.action(space, value, None)

  assert set(mapped_space.keys()) == {'other', 'a_first', 'a_second'}
  assert mapped_space['a_first'].labels == (('x', 'z'),)
  np.testing.assert_allclose(action['a'], [1.0, 2.0, 3.0])
  np.testing.assert_allclose(action['other'], [4.0, 5.0])


def test_reduce_observation_filters_reorders_and_removes_keys():
  space = core.Dict({'a': box(-1.0, 1.0, shape=(3,)), 'b': box(-1.0, 1.0), 'c': box(-1.0, 1.0)})
  transform = transforms.ReduceObservation(filter={'a': [2, 0], 'b': False, 'c': True})
  value = collections.OrderedDict(a=jnp.array([1.0, 2.0, 3.0]), b=jnp.zeros(2), c=jnp.ones(2))

  reduced_space = transform.observation_space(space, None)
  observation, _ = transform.observation(space, value, None)

  assert list(reduced_space.keys()) == ['a', 'c']
  np.testing.assert_allclose(observation['a'], [3.0, 1.0])
  assert 'b' not in observation


# ==================================================================================================
# Controllers


def make_pd(**kwargs):
  """A PD controller of two joints in [-1, 1], initialized."""
  controller = transforms.ProportionalDerivativeController(**kwargs)
  return initialize(controller, JOINT_SPACES, TORQUE_SPACE)


def test_pd_torque_follows_the_pd_law():
  controller, state = make_pd(actionable_pos='target', kp=(2.0, 3.0), kd=(0.5, 0.5))
  state = observe(controller, JOINT_SPACES, state, pos=[0.1, -0.2], vel=[1.0, -2.0])

  space = controller.action_space(TORQUE_SPACE, state)
  action, _ = controller.action(TORQUE_SPACE, {'target': jnp.array([0.5, 0.5])}, state)

  assert list(space.keys()) == ['target']
  np.testing.assert_allclose(action['joints_trq'], [2.0 * 0.4 - 0.5, 3.0 * 0.7 + 1.0], rtol=1e-6)


def test_pd_torque_is_clipped_to_the_torque_space():
  controller, state = make_pd(actionable_pos='target', kp=(100.0, 100.0), kd=(0.0, 0.0))
  state = observe(controller, JOINT_SPACES, state, pos=[-1.0, 1.0], vel=[0.0, 0.0])

  action, _ = controller.action(TORQUE_SPACE, {'target': jnp.array([1.0, -1.0])}, state)

  np.testing.assert_allclose(action['joints_trq'], [10.0, -10.0])


def test_pd_target_space_is_the_joint_range_widened_by_pos_margin():
  controller, state = make_pd(actionable_pos='target', kp=1.0, kd=1.0, pos_margin=0.1)

  space = controller.action_space(TORQUE_SPACE, state)

  np.testing.assert_allclose(space['target'].low, [-1.2, -1.2], rtol=1e-6)
  np.testing.assert_allclose(space['target'].high, [1.2, 1.2], rtol=1e-6)


def test_pd_actionable_kp_space_is_bounded_by_kp():
  controller, state = make_pd(actionable_pos='target', actionable_kp='gain', kp=(4.0, 5.0), kd=1.0)

  space = controller.action_space(TORQUE_SPACE, state)

  np.testing.assert_allclose(space['gain'].low, [0.0, 0.0])
  np.testing.assert_allclose(space['gain'].high, [4.0, 5.0])


def test_additive_pd_adds_to_the_incoming_torque_and_keeps_it_in_the_space():
  controller, state = make_pd(kp_passive=(1.0, 1.0), undamped=True, kp=0.0, additive_trq=True)
  state = observe(controller, JOINT_SPACES, state, pos=[0.5, -0.5], vel=[0.0, 0.0])

  space = controller.action_space(TORQUE_SPACE, state)
  action, _ = controller.action(TORQUE_SPACE, {'joints_trq': jnp.array([1.0, 1.0])}, state)

  assert list(space.keys()) == ['joints_trq']
  np.testing.assert_allclose(action['joints_trq'], [0.5, 1.5])


def make_spring(**kwargs):
  """A spring-muscle controller of two joints in [-1, 1], initialized."""
  controller = transforms.SpringMuscleController(
    poseq=(0.0, 0.2),
    keq=(2.0, 3.0),
    trq_max_flx=(1.0, 2.0),
    trq_max_ext=(1.5, 1.0),  # Asymmetric.
    **kwargs,
  )
  return initialize(controller, JOINT_SPACES, TORQUE_SPACE)


def spring_torque(controller, state, pos, flx, ext) -> np.ndarray:
  """The controller's torque at positions `pos` (at rest) for activations `flx` and `ext`."""
  state = state.replace(local=state.local.clone())  # Keep the caller's state usable.
  state = observe(controller, JOINT_SPACES, state, pos=pos, vel=[0.0, 0.0])
  value = collections.OrderedDict(muscle_flx=jnp.asarray(flx), muscle_ext=jnp.asarray(ext))
  action, _ = controller.action(TORQUE_SPACE, value, state)
  return np.asarray(action['joints_trq'])


def test_spring_muscle_rests_at_poseq_with_full_cocontraction():
  controller, state = make_spring()

  torque = spring_torque(controller, state, pos=[0.0, 0.2], flx=[1.0, 1.0], ext=[1.0, 1.0])

  np.testing.assert_allclose(torque, [0.0, 0.0], atol=1e-6)


def test_spring_muscle_stiffness_at_full_cocontraction_is_keq():
  controller, state = make_spring()

  low = spring_torque(controller, state, pos=[-0.1, 0.1], flx=[1.0, 1.0], ext=[1.0, 1.0])
  high = spring_torque(controller, state, pos=[0.1, 0.3], flx=[1.0, 1.0], ext=[1.0, 1.0])

  np.testing.assert_allclose((high - low) / 0.2, [-2.0, -3.0], rtol=1e-5)


def test_spring_muscle_single_muscle_gives_its_max_torque_at_the_far_limit():
  controller, state = make_spring()

  flexor = spring_torque(controller, state, pos=[1.0, 1.0], flx=[1.0, 1.0], ext=[0.0, 0.0])
  extensor = spring_torque(controller, state, pos=[-1.0, -1.0], flx=[0.0, 0.0], ext=[1.0, 1.0])

  np.testing.assert_allclose(flexor, [-1.0, -2.0], rtol=1e-5)
  np.testing.assert_allclose(extensor, [1.5, 1.0], rtol=1e-5)


def test_spring_muscle_replaces_torque_with_activations_in_the_action_space():
  controller, state = make_spring()

  space = controller.action_space(TORQUE_SPACE, state)

  assert list(space.keys()) == ['muscle_flx', 'muscle_ext']
  np.testing.assert_allclose(space['muscle_flx'].low, [0.0, 0.0])
  np.testing.assert_allclose(space['muscle_flx'].high, [1.0, 1.0])


def test_spring_muscle_torque_is_clipped_to_the_torque_space():
  controller = transforms.SpringMuscleController(
    poseq=0.0,
    keq=100.0,
    trq_max_flx=100.0,
    trq_max_ext=100.0,
  )
  controller, state = initialize(controller, JOINT_SPACES, TORQUE_SPACE)

  torque = spring_torque(controller, state, pos=[-1.0, 1.0], flx=[0.0, 1.0], ext=[1.0, 0.0])

  np.testing.assert_allclose(torque, [10.0, -10.0])
