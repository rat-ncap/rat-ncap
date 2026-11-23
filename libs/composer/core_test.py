import collections

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from composer import core

# ==================================================================================================
# Spaces


def test_box_vectorize_then_unvectorize_restores_space():
  box = core.Box(low=jnp.array([-1.0, 0.0]), high=jnp.array([1.0, 2.0]), labels=['joint'])

  restored = box.vectorize(3).unvectorize()

  assert restored.shape == box.shape
  assert restored.labels == box.labels
  np.testing.assert_array_equal(restored.low, box.low)
  np.testing.assert_array_equal(restored.high, box.high)


def test_box_vectorize_prepends_axis_with_tiled_bounds():
  box = core.Box(low=jnp.array([-1.0, 0.0]), high=jnp.array([1.0, 2.0]), labels=['joint'])

  vectorized = box.vectorize(3, label='world')

  assert vectorized.shape == (3, 2)
  assert vectorized.labels == ('world', 'joint')
  np.testing.assert_array_equal(vectorized.low, jnp.tile(box.low, (3, 1)))
  np.testing.assert_array_equal(vectorized.high, jnp.tile(box.high, (3, 1)))


def test_box_vectorize_generates_unused_axis_label():
  box = core.Box(low=0.0, high=1.0, shape=(2,), labels=['vectorize0'])

  vectorized = box.vectorize(3)

  assert vectorized.labels == ('vectorize1', 'vectorize0')


def test_box_vectorize_accepts_per_index_label_sequence():
  box = core.Box(low=0.0, high=1.0, shape=(2,), labels=['joint'])

  vectorized = box.vectorize(2, label=['left', 'right'])

  assert vectorized.labels == (('left', 'right'), 'joint')


@pytest.mark.parametrize(
  'label',
  [['left'], ['left', 'left']],
  ids=['wrong_length', 'duplicate_names'],
)
def test_box_vectorize_rejects_invalid_per_index_labels(label):
  box = core.Box(low=0.0, high=1.0, shape=(2,), labels=['joint'])

  with pytest.raises(AssertionError, match='exactly once'):
    box.vectorize(2, label=label)


def test_box_vectorize_rejects_existing_axis_label():
  box = core.Box(low=0.0, high=1.0, shape=(2,), labels=['joint'])

  with pytest.raises(AssertionError, match='already exists'):
    box.vectorize(3, label='joint')


def test_box_vectorize_labels_scalar_space_with_empty_labels():
  box = core.Box(low=0.0, high=1.0, shape=(), labels=[])

  vectorized = box.vectorize(3)

  assert vectorized.labels == ('vectorize0',)
  assert vectorized.unvectorize().labels == ()


def test_box_vectorize_keeps_unlabeled_space_unlabeled():
  box = core.Box(low=0.0, high=1.0, shape=(2,))

  assert box.vectorize(3).labels is None
  assert box.vectorize(3).unvectorize().labels is None


@pytest.mark.parametrize('mode', ['random', 'zero', 'min', 'max', 'mid'])
def test_box_sample_stays_within_bounds(mode):
  box = core.Box(low=jnp.array([-1.0, 0.5]), high=jnp.array([1.0, 2.0]))

  sample = box.sample(jax.random.key(0), mode=mode)

  box.validate(sample)
  assert bool(jnp.all((box.low <= sample) & (sample <= box.high)))


@pytest.mark.parametrize(
  'value',
  [jnp.zeros(3), jnp.zeros(2, dtype=jnp.int32)],
  ids=['shape', 'dtype'],
)
def test_box_validate_rejects_mismatched_value(value):
  box = core.Box(low=0.0, high=1.0, shape=(2,))

  with pytest.raises(ValueError, match='mismatch'):
    box.validate(value)


def test_dict_sample_draws_each_key_with_its_own_rng():
  space = core.Dict({'a': core.Box(0.0, 1.0, shape=(4,)), 'b': core.Box(0.0, 1.0, shape=(4,))})

  sample = space.sample(jax.random.key(0))

  assert list(sample.keys()) == ['a', 'b']
  assert not bool(jnp.array_equal(sample['a'], sample['b']))
  space.validate(sample)


def test_dict_validate_rejects_missing_key():
  space = core.Dict({'a': core.Box(0.0, 1.0, shape=(1,)), 'b': core.Box(0.0, 1.0, shape=(1,))})

  with pytest.raises(KeyError, match='Key mismatch'):
    space.validate({'a': jnp.zeros(1)})


# ==================================================================================================
# Values


def test_tile_value_lays_out_copies_population_major():
  value = {'command': jnp.array([1.0, 2.0])}  # [subtasks]

  tiled = core.tile_value(value, 3)

  np.testing.assert_array_equal(tiled['command'], [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])


# ==================================================================================================
# State


class Counter(eqx.Module):
  index: eqx.nn.StateIndex


def make_state(frozen: bool = True) -> tuple[eqx.nn.StateIndex, core.State]:
  model = core.initialize_state_index(Counter(eqx.nn.StateIndex(jnp.zeros(2))))
  return model.index, core.State(model, frozen=frozen)


def test_state_set_returns_new_state_and_invalidates_old():
  index, state = make_state()

  new_state = state.set(index, jnp.ones(2))

  np.testing.assert_array_equal(new_state.get(index), jnp.ones(2))
  with pytest.raises(ValueError, match='stale'):
    state.get(index)


def test_state_clone_keeps_values_after_original_is_set():
  index, state = make_state()

  clone = state.clone()
  state.set(index, jnp.ones(2))

  np.testing.assert_array_equal(clone.get(index), jnp.zeros(2))


def test_frozen_state_set_rejects_shape_change():
  index, state = make_state(frozen=True)

  with pytest.raises(ValueError, match='frozen'):
    state.set(index, jnp.zeros(3))


def test_unfrozen_state_set_accepts_shape_change():
  index, state = make_state(frozen=False)

  new_state = state.set(index, jnp.zeros(3))

  assert new_state.get(index).shape == (3,)


def test_state_passes_through_jit_with_values_and_frozen_flag():
  index, state = make_state(frozen=True)

  out = jax.jit(lambda s: s)(state)

  np.testing.assert_array_equal(out.get(index), jnp.zeros(2))
  with pytest.raises(ValueError, match='frozen'):
    out.set(index, jnp.zeros(3))


def test_state_rejects_index_without_string_marker():
  model = Counter(eqx.nn.StateIndex(jnp.zeros(2)))

  with pytest.raises(ValueError, match='initialize_state_index'):
    core.State(model)


def test_state_rejects_index_with_deleted_init():
  model = core.initialize_state_index(Counter(eqx.nn.StateIndex(jnp.zeros(2))))

  with pytest.raises(ValueError, match='already deleted'):
    core.State(eqx.nn.delete_init_state(model))


def test_initialize_state_index_makes_states_interchangeable_across_instances():
  model_a = core.initialize_state_index(Counter(eqx.nn.StateIndex(jnp.zeros(2))))
  model_b = core.initialize_state_index(Counter(eqx.nn.StateIndex(jnp.zeros(2))))

  state = core.State(model_a).set(model_a.index, jnp.ones(2))

  np.testing.assert_array_equal(state.get(model_b.index), jnp.ones(2))


# ==================================================================================================
# Entity


class EnvState(core.BaseEnvState):
  pass


def make_env_state() -> EnvState:
  return EnvState(
    rng=jax.random.key(0),
    steps=jnp.array(0),
    terminated=jnp.array(False),
    local=core.State({}),
  )


def make_box() -> core.Box:
  return core.Box(low=-jnp.inf, high=jnp.inf, shape=(1,))


class Add(core.Transform):
  """Adds `amount` to every action and observation value."""
  amount: float

  def __init__(self, amount: float):
    super().__init__()
    self.amount = amount

  def action(self, space, value, state):
    return collections.OrderedDict({k: v + self.amount for k, v in value.items()}), state

  def observation(self, space, value, state):
    return collections.OrderedDict({k: v + self.amount for k, v in value.items()}), state


class Scale(core.Transform):
  """Multiplies every action and observation value by `factor`."""
  factor: float

  def __init__(self, factor: float):
    super().__init__()
    self.factor = factor

  def action(self, space, value, state):
    return collections.OrderedDict({k: v * self.factor for k, v in value.items()}), state

  def observation(self, space, value, state):
    return collections.OrderedDict({k: v * self.factor for k, v in value.items()}), state


class RecordReset(core.Transform):
  """Appends its `name` to the shared `log` when reset."""
  name: str
  log: list

  def __init__(self, name: str, log: list):
    super().__init__()
    self.name = name
    self.log = log

  def reset(self, state):
    self.log.append(self.name)
    return state


class Probe(core.Entity):
  """An entity with one action and one observation key, `x`, that records the actions it receives."""
  received: list
  reading: float

  def __init__(self, children=None, transforms=(), reading: float = 0.0):
    super().__init__(children=children or {}, transforms=transforms)
    self.received = []
    self.reading = reading

  def action(self, state, action):
    self.received.append(action['x'])
    return state

  def action_space(self, state):
    return core.Dict({'x': make_box()})

  def observation(self, state):
    return collections.OrderedDict({'x': jnp.full(1, self.reading)}), state

  def observation_space(self, state):
    return core.Dict({'x': make_box()})


def test_entity_action_applies_transforms_outer_to_inner():
  # Transforms are listed innermost first: the action passes through Scale, then Add.
  probe = Probe(transforms=[Add(1.0), Scale(10.0)])

  core.entity_action(probe, make_env_state(), {'x': jnp.array([2.0])})

  np.testing.assert_array_equal(probe.received[0], [21.0])


def test_entity_observation_applies_transforms_inner_to_outer():
  probe = Probe(transforms=[Add(1.0), Scale(10.0)], reading=2.0)

  observation, _ = core.entity_observation(probe, make_env_state())

  np.testing.assert_array_equal(observation['x'], [30.0])


def test_entity_action_routes_prefixed_keys_to_children():
  child = Probe()
  parent = Probe(children={'leg': child})

  core.entity_action(parent, make_env_state(), {'x': jnp.array([1.0]), 'leg/x': jnp.array([2.0])})

  np.testing.assert_array_equal(parent.received[0], [1.0])
  np.testing.assert_array_equal(child.received[0], [2.0])


def test_entity_spaces_and_observation_prefix_child_keys():
  parent = Probe(children={'leg': Probe(reading=3.0)})
  state = make_env_state()

  observation, _ = core.entity_observation(parent, state)

  assert list(core.entity_action_space(parent, state).keys()) == ['x', 'leg/x']
  assert list(core.entity_observation_space(parent, state).keys()) == ['x', 'leg/x']
  np.testing.assert_array_equal(observation['leg/x'], [3.0])


def test_entity_reset_resets_own_transforms_then_children_depth_first():
  log = []
  leaf = Probe(transforms=[RecordReset('leaf', log)])
  child = Probe(children={'leaf': leaf}, transforms=[RecordReset('child', log)])
  sibling = Probe(transforms=[RecordReset('sibling', log)])
  parent = Probe(
    children={'child': child, 'sibling': sibling},
    transforms=[RecordReset('parent_inner', log), RecordReset('parent_outer', log)],
  )

  core.entity_reset(parent, make_env_state())

  assert log == ['parent_inner', 'parent_outer', 'child', 'leaf', 'sibling']


def test_entity_action_rejects_missing_key():
  probe = Probe()

  with pytest.raises(KeyError, match='Key missing'):
    core.entity_action(probe, make_env_state(), {})


@pytest.mark.parametrize(
  'children',
  [{'a/b': Probe()}, {'x': Probe()}],
  ids=['slash_in_child_name', 'child_name_collides_with_key'],
)
def test_entity_initialize_rejects_invalid_child_names(children):
  parent = Probe(children=children)

  with pytest.raises(KeyError):
    core.entity_initialize(parent, make_env_state(), {})
