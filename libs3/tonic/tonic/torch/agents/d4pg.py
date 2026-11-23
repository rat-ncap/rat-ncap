import typing as T

import torch

from tonic import core
from tonic.torch import agent, explorations, models, replays, updaters


# ==================================================================================================
# Model

def d4pg_default_model():
  return models.ActorCriticWithTargets(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.DeterministicPolicyHead(),
    ),
    critic=models.Critic(
      encoder=models.ObservationActionEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      # These value limits are for DeepMind Control Suite with 0.99 discount,
      # calculated by summing the geometric series:
      #   vmax = max_reward * (1 - discount ** max_length) / (1 - discount)
      #        = 1 * (1 - 0.99 ** 1000) / (1 - 0.99)
      #        = ~100
      #   vmin = -vmax
      # According to https://github.com/google-deepmind/acme/issues/57:
      # "Historically, a value of 150 was used to give it a bit of slack."
      # "The instantaneous rewards are bound in (-1, 1)". This is wrong, as
      # the rewards are bound in [0, 1], so negative value is impossible.
      # But we keep these value limits to be consistent with Tonic's benchmarking.
      head=models.DistributionalValueHead(-150., 150., 51),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )


# ==================================================================================================
# Replay

D4PGKeys = replays.OffPolicyKeys
D4PGData = replays.OffPolicyData
D4PGStep = replays.OffPolicyStep


# ==================================================================================================
# Agent

class D4PG(agent.Agent):
  """Distributed Distributional Deterministic Deep Policy Gradient.
  https://arxiv.org/pdf/1804.08617.pdf
  """
  model: models.ActorCriticWithTargets

  def __init__(
    self,
    model: models.ActorCriticWithTargets | None = None,
    exploration: explorations.NormalActionNoise | None = None,
    actor_updater: updaters.DistributionalDeterministicPolicyGradient | None = None,
    critic_updater: updaters.DistributionalDeterministicQLearning | None = None,
    # Dataset.
    replay_samples: int = 1_000_000,
    warmup_samples: int = 10_000,
    rollout_steps: int = 1,
    minibatch_samples: int = 128,
    minibatch_iterations: int = 50,
    # Returns.
    discount_factor: float = 0.99,
    return_steps: int = 5,
  ):
    self.model = model or d4pg_default_model()
    self.exploration = exploration or explorations.NormalActionNoise(warmup_samples=warmup_samples)
    self.actor_updater = actor_updater or updaters.DistributionalDeterministicPolicyGradient()
    self.critic_updater = critic_updater or updaters.DistributionalDeterministicQLearning()
    self.replay = replays.OffPolicyReplay[D4PGKeys, D4PGData, D4PGStep](
      max_samples=replay_samples,
      discount_factor=discount_factor,
      return_steps=return_steps,
    )
    self.replay_samples = replay_samples
    self.warmup_samples = warmup_samples
    self.rollout_steps = rollout_steps
    self.minibatch_samples = minibatch_samples
    self.minibatch_iterations = minibatch_iterations
    self.last_update_step = 0

  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    super().initialize(observation_space, action_space, seed)
    self.model.initialize(observation_space, action_space)
    self.replay.initialize(seed)
    self.exploration.initialize(lambda obs: self.model.actor(obs), action_space, seed)
    self.actor_updater.initialize(self.model)
    self.critic_updater.initialize(self.model)

  @T.override
  def step(self, observations: core.Observation) -> core.Action:
    # Get actions with exploration noise for training.
    with torch.no_grad():
      return self.exploration(observations, self.replay.cumulative_samples())

  @T.override
  def test_step(self, observations: core.Observation) -> core.Action:
    # Greedy actions for testing.
    with torch.no_grad():
      return self.model.actor(observations)
    
  @T.override
  def record(
    self,
    observations: core.Observation,
    actions: core.Action,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    next_observations: core.Observation,
  ) -> None:
    # Record transition in the replay.
    self.replay.record({
      'observations': observations,
      'actions': actions,
      'rewards': rewards,
      'terminations': terminations,
      'truncations': truncations,
      'next_observations': next_observations,
    })

    # Record transition in the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.record(observations)  # type: ignore
    if self.model.return_normalizer:
      self.model.return_normalizer.record(rewards)
    
    # Record resets in exploration.
    self.exploration.record(terminations | truncations)

  @T.override
  def update(self) -> dict[str, T.Any]:
    # Skip update if not enough samples collected or rollout not complete.
    if self.replay.cumulative_samples() < self.warmup_samples: return {}
    if self.replay.cumulative_steps() < self.last_update_step + self.rollout_steps: return {}
    
    # Mark this update step.
    self.last_update_step = self.replay.cumulative_steps()

    # Update both the actor and critic multiple times.
    actor_metrics: list[dict[str, torch.Tensor]] = []
    critic_metrics: list[dict[str, torch.Tensor]] = []
    for i in range(self.minibatch_iterations):
      keys = ('observations', 'actions', 'next_observations', 'rewards', 'discounts')
      minibatch = self.replay.get_minibatch(*keys, size=self.minibatch_samples)
      # Update the critic first.
      critic_metrics.append(self.critic_updater(
        observations=minibatch['observations'],
        actions=minibatch['actions'],
        next_observations=minibatch['next_observations'],
        rewards=minibatch['rewards'],
        discounts=minibatch['discounts'],
      ))
      # Update the actor using the new critic.
      actor_metrics.append(self.actor_updater(
        observations=minibatch['observations'],
      ))
      # Update the target networks.
      self.model.update_targets()

    # Collect training metrics.
    metrics = {f'actor/{k}': v for k, v in core.stack_tensors(actor_metrics).items()}
    metrics.update({f'critic/{k}': v for k, v in core.stack_tensors(critic_metrics).items()})

    # Update the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.update()
    if self.model.return_normalizer:
      self.model.return_normalizer.update()
    
    return metrics
