import typing as T

import torch

from tonic import core
from tonic.torch import agent, models, replays, updaters


# ==================================================================================================
# Model

def esrl_default_model():
  return models.ActorCritic(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP((64, 64), torch.nn.Tanh),
      head=models.DeterministicPolicyHead(),
    ),
    critic=models.Critic(
      encoder=models.ObservationEncoder(),
      torso=models.MLP((64, 64), torch.nn.Tanh),
      head=models.ValueHead(),
    ),
    observation_normalizer=models.MeanStdNormalizer(),
  )

# ==================================================================================================
# Replay

ESRLKeys = replays.OnPolicyKeys


ESRLData = replays.OnPolicyData


ESRLStep = replays.OnPolicyStep


# ==================================================================================================
# Agent

class ESRL(agent.Agent):
  """Evolution Strategy Reinforcement Learning. A counterpart to ES using critic-derived scoring."""
  model: models.ActorCritic

  def __init__(
    self,
    model: models.ActorCritic | None = None,
    critic_updater: updaters.VRegression | None = None,
    # Dataset.
    rollout_steps: int = 32,
    minibatch_samples: int = 4096,
    minibatch_iterations: int = 10,
    # Returns.
    discount_factor: float = 0.99,
    trace_decay: float = 0.,
    normalize_advantages: bool = False,
    recompute_advantages: bool = False,
    scores_key: T.Literal['advantages', 'returns', 'episodic_returns'] = 'advantages',
  ):
    """
    Args:
      model: Actor-critic container.
      critic_updater: Critic updater.
      rollout_steps: The number of steps for which to instantiate the replay buffer.
      minibatch_samples: The samples per replay buffer minibatch used to update the critic.
      minibatch_iterations: The iterations over the replay buffer during a critic update.
      discount_factor: The discount factor for the value function.
      trace_decay: The lambda value used to calculate lambda returns.
      normalize_advantages: Normalize advantages to a standard distribution.
      recompute_advantages: Recompute advantages at each critic update iteration.
      scores_key: The key corresponding to the data that should be used for model scoring.
    """
    self.model = model or esrl_default_model()
    
    self.critic_updater = critic_updater or updaters.VRegression()
    self.replay = replays.OnPolicyReplay[ESRLKeys, ESRLData, ESRLStep](
      max_steps=rollout_steps,
      discount_factor=discount_factor,
      trace_decay=trace_decay,
    )
    self.rollout_steps = rollout_steps
    self.minibatch_samples = minibatch_samples
    self.minibatch_iterations = minibatch_iterations
    self.discount_factor = discount_factor
    self.trace_decay = trace_decay
    self.normalize_advantages = normalize_advantages
    self.recompute_advantages = recompute_advantages
    self.scores_key: T.Literal['advantages', 'returns', 'episodic_returns'] = scores_key

  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    super().initialize(observation_space, action_space, seed)
    self.model.initialize(observation_space, action_space)
    self.replay.initialize(seed)
    self.critic_updater.initialize(self.model)

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
  
  @T.override
  def update(self) -> dict[str, T.Any]:
    # Skip update if replay is not full.
    assert self.replay.is_full()

    # Update the critic multiple times.
    critic_metrics: list[dict[str, torch.Tensor]] = []
    for i in range(self.minibatch_iterations):
      # Compute the lambda-returns over the full buffer. Must do this before splitting into 
      # shuffled minibatches because sequential ordering is assumed.
      if i == 0 or self.recompute_advantages:
        full = self.replay.get_full('observations', 'next_observations')
        with torch.no_grad():
          values = self.model.critic(full['observations'])
          next_values = self.model.critic(full['next_observations'])
        self.replay.compute_returns(values, next_values)
        self.replay.compute_advantages(normalize=self.normalize_advantages)
    
      minibatch_keys = ('observations', 'actions', 'advantages', 'returns')
      for minibatch in self.replay.get_minibatches(*minibatch_keys, size=self.minibatch_samples):
        # Update the critic.
        with torch.enable_grad():
          critic_metrics.append(self.critic_updater(
            observations=minibatch['observations'],
            returns=minibatch['returns'],
          ))

    # Collect training metrics.
    metrics: dict[str, int | torch.Tensor] = {
      f'critic/{k}': v for k, v in core.stack_tensors(critic_metrics).items()  # type: ignore
    }
    metrics.update(self.replay.get_full('advantages', 'returns'))  # type: ignore
    with torch.no_grad():
      match self.scores_key:
        case 'advantages' | 'returns':
          # Mean of per-timestep advantages or lambda-returns.
          scores = T.cast(torch.Tensor, metrics[self.scores_key])  # [num_steps, num_envs]
          metrics['scores'] = scores.reshape((self.rollout_steps, -1)).mean(0)  # [num_envs]
        
        case 'episodic_returns':
          # Mean of per-rollout episodic-returns.
          full = self.replay.get_full(
            'values', 'next_values', 'rewards', 'terminations', 'truncations'
          )
          scores = replays.episodic_returns(
            values=full['values'].reshape((self.rollout_steps, -1)),
            next_values=full['next_values'].reshape((self.rollout_steps, -1)),
            rewards=full['rewards'].reshape((self.rollout_steps, -1)),
            terminations=full['terminations'].reshape((self.rollout_steps, -1)),
            truncations=full['truncations'].reshape((self.rollout_steps, -1)),
            discount_factor=self.discount_factor,
          )
          metrics['scores'] = scores  # [num_envs]
        
    # Reset the replay for the next rollout.
    self.replay.reset()

    # Update the normalizers.
    if self.model.observation_normalizer:
      self.model.observation_normalizer.update()
    if self.model.return_normalizer:
      self.model.return_normalizer.update()
    
    return metrics
