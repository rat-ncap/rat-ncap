import collections
import types
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
import mjx_objective as mjxo
import policy
import tonic.torch.models as ttm

OBSERVATION_SPACE = cx.Dict(dict(target_vx=cx.Box(-1., 1., shape=(1,))))
ACTION_SPACE = cx.Dict(dict(act=cx.Box(-1., 1., shape=(2,))))
AGENT_CONFIG = oc.OmegaConf.create(dict(
  _target_='tonic.torch.agents.ES',
  model=dict(_target_='policy_test.make_model'),
  rollout_steps=1,
))


def make_model() -> ttm.ActorOnly:
  """Returns a small MLP actor with an observation normalizer (whose statistics are buffers)."""
  return ttm.ActorOnly(
    actor=ttm.Actor(
      encoder=ttm.ObservationEncoder(),
      torso=ttm.MLP((4,), torch.nn.Tanh),
      head=ttm.DeterministicPolicyHead(activation=torch.nn.Tanh),
    ),
    observation_normalizer=ttm.MeanStdNormalizer(),
  )


def make_agent(seed: int = 0) -> mjxo.Agent:
  """Returns an agent of `AGENT_CONFIG` initialized for the spaces with the seed."""
  torch.manual_seed(seed)
  agent = hydra.utils.instantiate(AGENT_CONFIG, _convert_='object')
  agent.initialize(cxg.to_gym_space(OBSERVATION_SPACE), cxg.to_gym_space(ACTION_SPACE), seed)
  return agent


def make_checkpoint(agent: mjxo.Agent, epoch: int = 0) -> mjxo.MjxEnvObjectiveCheckpoint:
  """Returns a checkpoint of the agent's model."""
  return mjxo.MjxEnvObjectiveCheckpoint(
    epoch=epoch,
    seed=0,
    env_config=oc.OmegaConf.create(dict(name='env')),
    agent_config=AGENT_CONFIG,
    subtask_space=OBSERVATION_SPACE,
    observation_space=OBSERVATION_SPACE,
    action_space=ACTION_SPACE,
    model_state_dict=agent.model.state_dict(),
  )


def get_observation(values: T.Sequence[float]) -> T.OrderedDict[str, jax.Array]:
  """Returns the observation of one world per value."""
  return collections.OrderedDict(target_vx=jnp.array(values, dtype=jnp.float32)[:, None])


def make_ready_policy(
  policy_cls: type[policy.EnsembledPolicy] = policy.EnsembledPolicy,
  *,
  population: int,
  worlds: int,
  **kwargs,
) -> policy.EnsembledPolicy:
  """Returns a policy of a checkpoint of `make_agent()`, initialized and reset for `worlds`."""
  ready = policy_cls(checkpoint=make_checkpoint(make_agent()), population=population, **kwargs)
  ready.initialize(
    OBSERVATION_SPACE.vectorize(worlds),
    ACTION_SPACE.vectorize(worlds),
    jax.random.split(jax.random.key(0), worlds),
  )
  ready.reset(jax.random.split(jax.random.key(1), worlds))
  return ready


# ==================================================================================================
# EnsembledPolicy


def test_every_specimen_acts_as_the_checkpoint_agent():
  agent = make_agent()
  values = [-0.5, 0.0, 0.5, 1.0]
  ensembled = make_ready_policy(population=2, worlds=len(values))

  actions = ensembled.step(get_observation(values), jax.random.split(jax.random.key(2), 4))

  for world, value in enumerate(values):
    expected = agent.test_step(collections.OrderedDict(target_vx=torch.tensor([[value]])))
    np.testing.assert_allclose(
      np.asarray(actions['act'][world]),
      expected['act'][0].numpy(),
      rtol=1e-6,
    )


def test_checkpoint_directory_is_relative_to_the_repository_root(tmp_path, monkeypatch):
  agent = make_agent()
  for epoch in (0, 10):
    mjxo.save_checkpoint(
      tmp_path,
      epoch=epoch,
      env_config=oc.OmegaConf.create(dict(name='env')),
      agent_config=AGENT_CONFIG,
      subtask_space=OBSERVATION_SPACE,
      observation_space=OBSERVATION_SPACE,
      action_space=ACTION_SPACE,
      agent=agent,
    )
  monkeypatch.setattr(policy, 'REPO_ROOT', tmp_path)

  loaded = policy.EnsembledPolicy(checkpoint='checkpoint', epoch='last')

  assert loaded.checkpoint.epoch == 10


def test_reset_rejects_worlds_that_are_not_a_multiple_of_the_population():
  ensembled = policy.EnsembledPolicy(checkpoint=make_checkpoint(make_agent()), population=2)

  with pytest.raises(ValueError, match='multiple of population'):
    ensembled.reset(jax.random.split(jax.random.key(0), 3))


def test_initialize_rejects_an_action_space_other_than_the_checkpoints():
  ensembled = policy.EnsembledPolicy(checkpoint=make_checkpoint(make_agent()))
  other_action_space = cx.Dict(dict(act=cx.Box(-1., 1., shape=(3,))))

  with pytest.raises(ValueError, match='action space'):
    ensembled.initialize(
      OBSERVATION_SPACE.vectorize(1),
      other_action_space.vectorize(1),
      jax.random.split(jax.random.key(0), 1),
    )


# ==================================================================================================
# PerturbedPolicy


def test_perturbation_is_shared_by_a_specimens_worlds_and_keeps_signs():
  original = make_agent().model.actor.state_dict()

  perturbed = make_ready_policy(
    policy.PerturbedPolicy,
    population=2,
    worlds=4,
    mode='multiplicative',
    sigma=0.5,
    include=('*',),
    exclude=(),
  )

  for name, param in perturbed.agent.params.items():
    torch.testing.assert_close(param[0], param[1])  # Specimen 0's two worlds.
    torch.testing.assert_close(param[2], param[3])  # Specimen 1's two worlds.
    assert not torch.equal(param[0], param[2])
    assert torch.all(param * original[name] >= 0)


def test_zero_sigma_leaves_the_parameters_unchanged():
  original = make_agent().model.actor.state_dict()

  perturbed = make_ready_policy(
    policy.PerturbedPolicy,
    population=2,
    worlds=2,
    mode='additive',
    sigma=0.,
    include=('*',),
    exclude=(),
  )

  for name, param in perturbed.agent.params.items():
    torch.testing.assert_close(param, original[name].expand_as(param))


# ==================================================================================================
# ScheduledObservation


def test_scheduled_observation_holds_each_value_from_its_start_time():
  schedule = policy.ScheduledObservation(key='target_vx', schedule=[(1.0, 0.4), (0.0, 0.2)])
  steps = jnp.array([0, 10, 33, 34])  # 0, 0.3, 0.99 and 1.02 s at 0.03 s per step.
  step = types.SimpleNamespace(
    state=types.SimpleNamespace(steps=steps, env_timestep=0.03),
    observation={'target_vx': jnp.zeros((4, 1))},
  )

  step = schedule(T.cast(cx.Step, step))

  np.testing.assert_allclose(np.asarray(step.observation['target_vx'][:, 0]), [0.2, 0.2, 0.2, 0.4])
