import typing as T

import torch

from tonic import core
from tonic.torch import agent, explorations, models, replays, updaters


# ==================================================================================================
# Model

def td3_default_model():
  return models.ActorTwinCriticWithTargets(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.DeterministicPolicyHead(),
    ),
    critic=models.Critic(
      encoder=models.ObservationActionEncoder(),
      torso=models.MLP((256, 256), torch.nn.ReLU),
      head=models.ValueHead(),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )


# ==================================================================================================
# Replay

TD3Keys = replays.OffPolicyKeys
TD3Data = replays.OffPolicyData
TD3Step = replays.OffPolicyStep


# ==================================================================================================
# Agent

class TD3(agent.Agent):
  """Twin Delayed Deep Deterministic Policy Gradient. https://arxiv.org/pdf/1802.09477.pdf"""
  model: models.ActorTwinCriticWithTargets

  def __init__(
    self,
    model: models.ActorTwinCriticWithTargets | None = None,
    exploration: explorations.NormalActionNoise | None = None,
    actor_updater: updaters.DeterministicPolicyGradient | None = None,
    critic_updater: updaters.TwinCriticDeterministicQLearning | None = None,
    # Dataset.
    replay_samples: int = 1_000_000,
    warmup_samples: int = 10_000,
    rollout_steps: int = 1,
    minibatch_samples: int = 128,
    minibatch_iterations: int = 50,
    delay_iterations: int = 2,
    # Returns.
    discount_factor: float = 0.99,
    return_steps: int = 1,
  ):
    self.model = model or td3_default_model()
    self.exploration = exploration or explorations.NormalActionNoise(warmup_samples=warmup_samples)
    self.actor_updater = actor_updater or updaters.DeterministicPolicyGradient()
    self.critic_updater = critic_updater or updaters.TwinCriticDeterministicQLearning()
    self.replay = replays.OffPolicyReplay[TD3Keys, TD3Data, TD3Step](
      max_samples=replay_samples,
      discount_factor=discount_factor,
      return_steps=return_steps,
    )
    self.replay_samples = replay_samples
    self.warmup_samples = warmup_samples
    self.rollout_steps = rollout_steps
    self.minibatch_samples = minibatch_samples
    self.minibatch_iterations = minibatch_iterations
    self.delay_iterations = delay_iterations
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
    self.exploration.record(truncations | terminations)

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
      # Update the actor using the new critic (delayed).
      if (i + 1) % self.delay_iterations == 0:
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
