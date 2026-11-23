import typing as T

import torch

from tonic import core
from tonic.torch import agent, models, replays


# ==================================================================================================
# Model

def es_default_model():
  return models.ActorOnly(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP((64, 64), torch.nn.Tanh),
      head=models.DeterministicPolicyHead(),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )

# ==================================================================================================
# Replay

ESKeys = T.Literal['rewards']


class ESData(T.TypedDict):
  rewards: torch.Tensor


class ESStep(T.TypedDict):
  rewards: torch.Tensor


# ==================================================================================================
# Agent

class ES(agent.Agent):
  """Evolution Strategy. Rollouts are scored by the average per-timestep reward."""
  model: models.ActorOnly

  def __init__(
    self,
    model: models.ActorOnly | None = None,
    rollout_steps: int = 32,
  ):
    self.model = model or es_default_model()
    self.rollout_steps = rollout_steps

    # Store the per-timestep rewards
    self.replay = replays.OnPolicyReplay[ESKeys, ESData, ESStep](max_steps=rollout_steps)

  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    super().initialize(observation_space, action_space, seed)
    self.model.initialize(observation_space, action_space)

  @T.override
  def step(self, observations: core.Observation) -> core.Action:
    return self.model.actor(observations)

  @T.override
  def test_step(self, observations: core.Observation) -> core.Action:
    # Sample actions for testing.
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
    self.replay.record({'rewards': rewards})

    # Record transition in the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.record(observations)  # type: ignore

  @T.override
  def update(self) -> dict[str, torch.Tensor]:
    # Collect training metrics.
    rewards = self.replay.get_full('rewards')['rewards']
    metrics = {'scores': rewards.reshape((self.rollout_steps, -1)).mean(0)}
    
    # Reset the replay for the next rollout.
    self.replay.reset()

    # Update the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.update()

    return metrics
