import collections
import pathlib
import pickle
import typing as T

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf as oc
import pytest
import torch

import composer as cx
import composer.utils.gym as cxg
import evolution2 as evolution
import mjx_objective as mjxo
import tonic.torch.models as ttm
import utils.logging as utlog

SUBTASKS: dict[str, T.Sequence[float]] = dict(target_vx=[0.0, 0.2, 0.4, 1.0])
ROLLOUT_STEPS = 3
POPULATION = 4
OBSERVATION_SPACE = cx.Dict(dict(target_vx=cx.Box(-1., 1., shape=(1,))))
ACTION_SPACE = cx.Dict(dict(act=cx.Box(-1., 1., shape=(2,))))
AGENT_CONFIG = oc.OmegaConf.create(dict(
  _target_='tonic.torch.agents.ES',
  model=dict(_target_='mjx_objective_test.make_model'),
  rollout_steps=ROLLOUT_STEPS,
))


def make_model() -> ttm.ActorOnly:
  """Returns a small MLP actor with an observation normalizer."""
  return ttm.ActorOnly(
    actor=ttm.Actor(
      encoder=ttm.ObservationEncoder(),
      torso=ttm.MLP((4,), torch.nn.Tanh),
      head=ttm.DeterministicPolicyHead(activation=torch.nn.Tanh),
    ),
    observation_normalizer=ttm.MeanStdNormalizer(),
  )


class TargetVelocityEnv:
  """A vectorized env that observes and is rewarded with its subtask's `target_vx`, plus Gaussian
  noise of standard deviation `noise` drawn from its reset keys. Its episodes never end."""

  def __init__(self, noise: float = 0.):
    self.noise = noise

  def initialize(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns the state of each world: its target velocity and random key."""
    return jnp.zeros(rng.shape[0]), rng

  def subtask_space(self, state: tuple[jax.Array, jax.Array]) -> cx.Dict:
    return cx.Dict(dict(target_vx=cx.Box(-1., 1., shape=()))).vectorize(len(state[0]))

  def observation_space(self, state: tuple[jax.Array, jax.Array]) -> cx.Dict:
    return OBSERVATION_SPACE.vectorize(len(state[0]))

  def action_space(self, state: tuple[jax.Array, jax.Array]) -> cx.Dict:
    return ACTION_SPACE.vectorize(len(state[0]))

  def reset(
    self,
    state: tuple[jax.Array, jax.Array],
    rng: jax.Array,
    *,
    reset: tuple[jax.Array, cx.Step] | None = None,
    subtask: dict[str, jax.Array] | None = None,
  ) -> cx.Step:
    assert subtask is not None
    return self.get_step(subtask['target_vx'], rng)

  def step(self, state: tuple[jax.Array, jax.Array], action: cx.Action) -> cx.Step:
    target_vx, rng = state
    return self.get_step(target_vx, rng)

  def get_step(self, target_vx: jax.Array, rng: jax.Array) -> cx.Step:
    """Returns the step of the worlds' target velocities, drawing the noise from their keys."""
    rng, noise_rng = jax.vmap(jax.random.split)(rng).transpose(1, 0)
    noise = self.noise * jax.vmap(jax.random.normal)(noise_rng)
    return cx.Step(
      state=(target_vx, rng),
      observation=collections.OrderedDict(target_vx=target_vx[:, None]),
      terminated=jnp.zeros(len(target_vx), dtype=bool),
      truncated=jnp.zeros(len(target_vx), dtype=bool),
      reward=target_vx + noise,
      metrics={},
    )


def make_objective(
  log_dir: pathlib.Path,
  noise: float = 0.,
  **kwargs,
) -> mjxo.MjxEnvObjective:
  """Returns an initialized objective on `TargetVelocityEnv`s, for `POPULATION` training and 2 test
  parameter vectors."""
  env_config = oc.OmegaConf.create(dict(
    _target_='mjx_objective_test.TargetVelocityEnv',
    noise=noise,
  ))
  objective = mjxo.MjxEnvObjective(
    train_env_config=env_config,
    test_env_config=env_config,
    agent_config=AGENT_CONFIG,
    logger=utlog.Logger(log_dir),
    subtasks=SUBTASKS,
    train_timeouts=ROLLOUT_STEPS,
    test_timeouts=ROLLOUT_STEPS,
    **kwargs,
  )
  objective.initialize(seed=0, train_size=POPULATION, test_size=2)
  return objective


def make_agent(seed: int) -> mjxo.Agent:
  """Returns an agent of `AGENT_CONFIG` initialized for the spaces with the seed."""
  torch.manual_seed(seed)
  agent = hydra.utils.instantiate(AGENT_CONFIG, _convert_='object')
  agent.initialize(cxg.to_gym_space(OBSERVATION_SPACE), cxg.to_gym_space(ACTION_SPACE), seed)
  return agent


def get_observations(count: int, value: float | None = None) -> T.OrderedDict[str, torch.Tensor]:
  """Returns `count` observations: of `value`, or distinct from -1 to 1 if None."""
  target_vx = torch.linspace(-1., 1., count) if value is None else torch.full((count,), value)
  return collections.OrderedDict(target_vx=target_vx[:, None])


# ==================================================================================================
# Checkpoint


@pytest.fixture
def checkpoint_dir(tmp_path: pathlib.Path) -> pathlib.Path:
  """A checkpoint directory with the models of epochs 0, 10 and 20, and `last.pt` of epoch 20. Each
  model's `epoch` tensor holds its epoch, and -1 in `last.pt`."""
  checkpoint_dir = tmp_path / 'checkpoint'
  checkpoint_dir.mkdir()
  oc.OmegaConf.save(oc.OmegaConf.create(dict(name='env')), checkpoint_dir / 'env_config.yaml')
  oc.OmegaConf.save(AGENT_CONFIG, checkpoint_dir / 'agent_config.yaml')
  for name in ('subtask_space', 'observation_space', 'action_space'):
    with open(checkpoint_dir / f'{name}.pkl', 'wb') as f:
      pickle.dump(OBSERVATION_SPACE, f)
  with open(checkpoint_dir / 'last_epoch.pkl', 'wb') as f:
    pickle.dump(20, f)
  for name, value in (('0', 0), ('10', 10), ('20', 20), ('last', -1)):
    torch.save(dict(epoch=torch.tensor(value)), checkpoint_dir / f'{name}.pt')
  return checkpoint_dir


@pytest.mark.parametrize(
  'epoch, has_last, expected_epoch, expected_value',
  [
    ('last', True, 20, -1),
    ('last', False, 20, 20),
    ('first', True, 0, 0),
    (10, True, 10, 10),
  ],
  ids=['last', 'last_without_last_file', 'first', 'numbered'],
)
def test_load_checkpoint_resolves_the_epoch(
  checkpoint_dir, epoch, has_last, expected_epoch, expected_value,
):
  if not has_last:
    (checkpoint_dir / 'last.pt').unlink()

  checkpoint = mjxo.load_checkpoint(checkpoint_dir, epoch=epoch)

  assert checkpoint.epoch == expected_epoch
  assert checkpoint.model_state_dict['epoch'].item() == expected_value


def test_load_checkpoint_rejects_an_unknown_epoch_name(checkpoint_dir):
  with pytest.raises(ValueError, match='middle'):
    mjxo.load_checkpoint(checkpoint_dir, epoch=T.cast(T.Any, 'middle'))


def test_load_checkpoint_raises_without_checkpoints(tmp_path):
  with pytest.raises(FileNotFoundError):
    mjxo.load_checkpoint(tmp_path, epoch='first')


def test_saved_checkpoint_rebuilds_an_agent_that_acts_the_same(tmp_path):
  agent = make_agent(seed=1)
  normalizer = agent.model.observation_normalizer
  assert normalizer is not None
  normalizer.record(get_observations(8))
  normalizer.update()
  observations = get_observations(5)

  mjxo.save_checkpoint(
    tmp_path,
    epoch=0,
    env_config=oc.OmegaConf.create(dict(name='env')),
    agent_config=AGENT_CONFIG,
    subtask_space=OBSERVATION_SPACE,
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    agent=agent,
  )
  loaded = mjxo.load_checkpoint(tmp_path / 'checkpoint', epoch='first').initialize_agent()

  torch.testing.assert_close(loaded.test_step(observations), agent.test_step(observations))


# ==================================================================================================
# EnsembledAgent


def test_ensemble_member_acts_as_an_agent_of_its_parameters():
  ensemble = mjxo.EnsembledAgent(lambda: hydra.utils.instantiate(AGENT_CONFIG, _convert_='object'))
  ensemble.initialize(cxg.to_gym_space(OBSERVATION_SPACE), cxg.to_gym_space(ACTION_SPACE), seed=0)
  params, buffers = ensemble.initial_state(3)
  other = make_agent(seed=1)
  with torch.no_grad():
    for name, param in T.cast(torch.nn.Module, other.model.actor).named_parameters():
      params[name][1] = param
  ensemble.set_state(params, buffers)
  ensemble.reset()

  actions = T.cast(dict[str, torch.Tensor], ensemble.test_step(get_observations(3, value=0.5)))

  expected = T.cast(dict[str, torch.Tensor], other.test_step(get_observations(1, value=0.5)))
  torch.testing.assert_close(actions['act'][1:2], expected['act'])
  assert not torch.equal(actions['act'][0], actions['act'][1])


# ==================================================================================================
# MjxEnvObjective


@pytest.mark.parametrize('mode', ['train', 'test'])
def test_score_is_the_mean_reward_per_step_averaged_over_the_subtasks(tmp_path, mode):
  subtask_indices = [1, 3]
  objective = make_objective(tmp_path, train_subtasks=subtask_indices)
  size = POPULATION if mode == 'train' else 2
  params = objective.get_initial_params(torch.arange(size))
  stats: dict[str, float | torch.Tensor] = {}

  scores = objective.get_score(params, torch.arange(size), mode=mode, stats=stats, files={})

  target_vx = np.array(SUBTASKS['target_vx'])[subtask_indices]
  torch.testing.assert_close(scores, torch.full((size,), target_vx.mean()))
  for index, value in zip(subtask_indices, target_vx, strict=True):
    assert stats[f'subtask_return/{index}'] == pytest.approx(value)


def test_score_is_determined_by_the_seeds(tmp_path):
  objective = make_objective(tmp_path, noise=0.1)
  params = objective.get_initial_params(torch.arange(2))

  first, again, other = (
    objective.get_score(params, seeds, mode='test', stats={}, files={})
    for seeds in (torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 3]))
  )

  torch.testing.assert_close(first, again)
  assert not torch.equal(first, other)


@pytest.mark.parametrize(
  'epoch, expected_names',
  [(500, ['500', 'last']), (510, ['last'])],
  ids=['checkpoint_epoch', 'other_epoch'],
)
def test_save_writes_last_and_periodic_checkpoints_of_the_first_tested_params(
  tmp_path, monkeypatch, epoch, expected_names,
):
  # The env's steps have no MuJoCo spec to save.
  monkeypatch.setattr(mjxo.cxr, 'save_steps', lambda path, steps: path.touch())
  objective = make_objective(tmp_path, checkpoint_freq=500)
  params = objective.get_initial_params(torch.arange(2))
  stats: dict[str, float | torch.Tensor] = {}
  files: dict[str, T.Any] = {}
  scores = objective.get_score(params, torch.arange(2), mode='test', stats=stats, files=files)
  first_params = {name: param[0].clone() for name, param in files['params'].items()}

  objective.on_save(
    epoch=epoch,
    algorithm=T.cast(T.Any, None),
    train_data=T.cast(T.Any, None),
    test_data=evolution.AlgorithmTestData(scores=scores, stats=stats, files=files),
    log_dir=tmp_path,
  )

  checkpoint_dir = tmp_path / 'checkpoint'
  assert sorted(path.stem for path in checkpoint_dir.glob('*.pt')) == expected_names
  assert sorted(path.stem for path in (tmp_path / 'rollout' / 'params').glob('*.pt')) == (
    expected_names
  )
  state_dict = torch.load(checkpoint_dir / 'last.pt')
  for name, param in first_params.items():
    torch.testing.assert_close(state_dict[f'actor.{name}'], param)
