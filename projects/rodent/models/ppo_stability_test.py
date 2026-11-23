import collections
import copy
import csv
import pathlib
import types
import typing as T

import gymnasium
import pytest
import torch

import tonic.logging
import tonic.torch.models as ttm
import tonic.torch.updaters as ttu

import composer.utils.gym as cxg
from models import ppo_stability

BATCH = 4  # Parallel environments.
OBSERVATION_SPACE = gymnasium.spaces.Dict({'obs': cxg.Box(-1., 1., shape=(3,))})
ACTION_SPACE = gymnasium.spaces.Dict({'act': cxg.Box(-1., 1., shape=(2,))})


def make_model(seed: int = 0) -> ttm.ActorCritic:
  """A tiny actor-critic with a Gaussian policy, initialized for the spaces."""
  torch.manual_seed(seed)
  model = ttm.ActorCritic(
    actor=ttm.Actor(
      encoder=ttm.ObservationEncoder(),
      torso=ttm.MLP((4,), torch.nn.Tanh),
      head=ttm.StochasticDetachedStdPolicyHead(),
    ),
    critic=ttm.Critic(
      encoder=ttm.ObservationEncoder(),
      torso=ttm.MLP((4,), torch.nn.Tanh),
      head=ttm.ValueHead(),
    ),
  )
  model.initialize(OBSERVATION_SPACE, ACTION_SPACE)
  return model


def make_updater(model: ttm.ActorCritic, **kwargs: T.Any) -> ppo_stability.DiagnosedClippedRatio:
  """A diagnosed actor updater of `model` (no logged steps unless given)."""
  kwargs.setdefault('logged_steps', 0)
  updater = ppo_stability.DiagnosedClippedRatio(**kwargs)
  updater.initialize(model)
  updater.begin_update(0)
  return updater


def make_agent(**kwargs) -> ppo_stability.RestartPPO:
  """A `RestartPPO` agent of the tiny model, with 2-step rollouts and one minibatch per update."""
  agent = ppo_stability.RestartPPO(
    model=make_model(),
    actor_updater=ppo_stability.DiagnosedClippedRatio(logged_steps=0),
    rollout_steps=2,
    minibatch_samples=2 * BATCH,
    minibatch_iterations=1,
    **kwargs,
  )
  agent.initialize(OBSERVATION_SPACE, ACTION_SPACE, seed=0)
  return agent


def batch(seed: int) -> collections.OrderedDict[str, torch.Tensor]:
  """Random observations of the parallel environments."""
  generator = torch.Generator().manual_seed(seed)
  return collections.OrderedDict(obs=torch.rand(BATCH, 3, generator=generator) * 2 - 1)


def actor(model: ttm.ActorCritic) -> torch.nn.Module:
  """The model's actor module."""
  return T.cast(torch.nn.Module, model.actor)


def fill_replay(agent: ppo_stability.RestartPPO, steps: int = 2, seed: int = 0) -> None:
  """Records `steps` random transitions with random rewards."""
  generator = torch.Generator().manual_seed(seed)
  for t in range(steps):
    observations = batch(seed * 10 + t)
    actions = agent.step(observations)
    agent.record(
      observations,
      actions,
      rewards=torch.randn(BATCH, generator=generator),
      terminations=torch.zeros(BATCH, dtype=torch.bool),
      truncations=torch.zeros(BATCH, dtype=torch.bool),
      next_observations=batch(seed * 10 + t + 1),
    )


def parameters(module: torch.nn.Module) -> list[torch.Tensor]:
  """Copies of the module's parameters."""
  return [param.detach().clone() for param in module.parameters()]


def changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
  """Whether any parameter of `module` differs from `before`."""
  return any(not torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))


def actor_batch(model: ttm.ActorCritic, ratios: list[float]):
  """Observations, the policy's actions, and log-probabilities that make the actions' probability
  ratios (new / rollout) equal to `ratios`."""
  observations = {'obs': torch.rand(len(ratios), 3, generator=torch.Generator().manual_seed(1))}
  with torch.no_grad():
    policy = model.actor(observations)
    actions = policy.sample()
    log_probs = policy.log_prob(actions).sum(-1) - torch.log(torch.tensor(ratios))
  return observations, actions, log_probs


@pytest.fixture
def log_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
  """A log directory for tonic's logger (as `tonic.Trainer` sets)."""
  monkeypatch.setattr(tonic.logging, 'getLogger', lambda: types.SimpleNamespace(log_dir=tmp_path))
  return tmp_path


# ==================================================================================================
# DiagnosedClippedRatio


def test_loss_and_clip_fraction_are_the_clipped_objective():
  model = make_model()
  updater = make_updater(model, ratio_clip=0.2)
  ratios = torch.tensor([0.5, 1., 1.5])
  advantages = torch.tensor([1., -1., 1.])
  observations, actions, log_probs = actor_batch(model, ratios.tolist())

  metrics = updater(observations, actions, advantages, log_probs)

  expected = -torch.min(ratios * advantages, ratios.clamp(0.8, 1.2) * advantages).mean()
  torch.testing.assert_close(metrics['loss'], expected)
  torch.testing.assert_close(metrics['clip_fraction'], torch.tensor(2 / 3))


def test_step_matches_clipped_ratio_without_rollback(log_dir):
  model = make_model()
  reference = copy.deepcopy(model)
  updater = make_updater(model, logged_steps=1)
  reference_updater = ttu.ClippedRatio()
  reference_updater.initialize(reference)
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])
  advantages = torch.tensor([1., -1., 0.5, 2.])

  metrics = updater(observations, actions, advantages, log_probs)
  reference_metrics = reference_updater(observations, actions, advantages, log_probs)

  for a, b in zip(actor(model).parameters(), actor(reference).parameters(), strict=True):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
  for key in ('loss', 'kl', 'clip_fraction'):
    torch.testing.assert_close(metrics[key], reference_metrics[key])


def test_split_step_parts_sum_to_the_adam_step():
  model = make_model()
  updater = make_updater(model)
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])
  advantages = torch.tensor([1., -1., 0.5, 2.])
  updater(observations, actions, advantages, log_probs)  # Warm Adam up.
  updater.optimizer.zero_grad()
  loss = -(model.actor(observations).log_prob(actions).sum(-1) * advantages).mean()
  loss.backward()
  before = parameters(actor(model))

  momentum, gradient = updater.split_step()
  updater.adam_step()

  for param, old, m, g in zip(actor(model).parameters(), before, momentum, gradient, strict=True):
    torch.testing.assert_close(param.detach() - old, m + g)


def test_rollback_restores_parameters_zeroes_momentum_and_stops():
  model = make_model()
  updater = make_updater(
    model, rollback=True, kl_threshold=-1., optimizer=lambda p: torch.optim.Adam(p, lr=0.1)
  )
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])
  before = parameters(actor(model))

  metrics = updater(observations, actions, torch.tensor([1., -1., 0.5, 2.]), log_probs)

  assert not changed(before, actor(model))
  assert all(not state['exp_avg'].any() for state in updater.optimizer.state.values())
  assert metrics['stop'] and metrics['rolled_back'] == 1


def test_reset_momentum_zeroes_the_momentum_at_each_update():
  model = make_model()
  updater = make_updater(model, reset_momentum=True)
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])
  updater(observations, actions, torch.tensor([1., -1., 0.5, 2.]), log_probs)

  updater.begin_update(1)

  assert all(not state['exp_avg'].any() for state in updater.optimizer.state.values())


def test_diagnostics_log_the_first_steps_of_each_update(log_dir):
  model = make_model()
  updater = make_updater(model, logged_steps=2)
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])

  for _ in range(3):
    updater(observations, actions, torch.tensor([1., -1., 0.5, 2.]), log_probs)

  with open(log_dir / 'actor_steps.csv') as f:
    rows = list(csv.DictReader(f))
  assert [int(row['step']) for row in rows] == [1, 2]
  assert {'kl_momentum', 'kl_gradient', 'top1pct_share', 'kl_from_rollout'} <= set(rows[0])


def test_diagnostics_skip_steps_at_learning_rate_zero(log_dir):
  model = make_model()
  updater = make_updater(model, logged_steps=2)
  updater.optimizer.param_groups[0]['lr'] = 0.
  observations, actions, log_probs = actor_batch(model, [0.9, 1., 1.1, 1.])

  updater(observations, actions, torch.tensor([1., -1., 0.5, 2.]), log_probs)

  assert not (log_dir / 'actor_steps.csv').exists()


# ==================================================================================================
# RestartPPO


def test_update_waits_for_a_full_replay():
  agent = make_agent()
  fill_replay(agent, steps=1)

  assert agent.update() == {}
  assert agent.updates == 0


def test_warmup_updates_keep_parameters_and_build_adam_state():
  agent = make_agent(warmup_updates=2)
  learning_rate = agent.actor_updater.optimizer.param_groups[0]['lr']
  before = parameters(agent.model)

  for update in range(2):
    fill_replay(agent, seed=update)
    agent.update()

    assert not changed(before, agent.model)
    assert agent.actor_updater.optimizer.param_groups[0]['lr'] == learning_rate
  optimizer_state = agent.actor_updater.optimizer.state.values()
  assert any(state['exp_avg_sq'].any() for state in optimizer_state)

  fill_replay(agent, seed=2)
  agent.update()

  assert changed(before, agent.model)


def test_init_checkpoint_loads_into_the_model(tmp_path: pathlib.Path):
  source = make_model(seed=1)
  torch.save(source.state_dict(), tmp_path / 'model.pt')

  agent = make_agent(init_checkpoint=str(tmp_path / 'model.pt'))

  for a, b in zip(agent.model.parameters(), source.parameters(), strict=True):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
