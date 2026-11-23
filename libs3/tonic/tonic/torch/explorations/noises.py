"""Exploration noise for continuous action spaces."""

import typing as T

import gymnasium.spaces
import torch

from tonic import core
from tonic.torch import space


class RandomPolicy:
  """Random policy that uniformly samples the action space for exploration."""

  def __init__(self, action_space: core.ActionSpace, rng: torch.Generator):
    if isinstance(action_space, gymnasium.spaces.Dict):
      assert all(isinstance(a, gymnasium.spaces.Box) for a in action_space.spaces.values())
    else:
      assert isinstance(action_space, gymnasium.spaces.Box)
    self._rng = rng
    self._action_space = action_space

  def __call__(self) -> core.Action:
    return space.sample_tensors(self._action_space, rng=self._rng)


class NoActionNoise:
  def __init__(self, warmup_samples: int = 20_000):
    self.warmup_samples = warmup_samples

  def initialize(
    self,
    policy: T.Callable[[core.Observation], core.Action],
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    if isinstance(action_space, gymnasium.spaces.Dict):
      assert all(isinstance(a, gymnasium.spaces.Box) for a in action_space.spaces.values())
    else:
      assert isinstance(action_space, gymnasium.spaces.Box)
    self._rng = torch.Generator().manual_seed(seed)
    self._action_space = action_space
    self._model_policy = policy
    self._random_policy = RandomPolicy(action_space, rng=self._rng)

  @torch.no_grad()
  def __call__(self, observations: core.Observation, samples: int) -> core.Action:
    if samples <= self.warmup_samples:
      return self._random_policy()
    
    actions = self._model_policy(observations)

    # Compute noises.
    if isinstance(actions, dict):
      return {
        k: self._compute_noise(self._action_space.spaces[k], v)  # type: ignore
        for k, v in actions.items()
      }
    else:
      return self._compute_noise(self._action_space, actions)  # type: ignore

  def record(self, resets: torch.Tensor) -> None:
    pass

  def _compute_noise(self, space: gymnasium.spaces.Box, action: torch.Tensor) -> torch.Tensor:
    low = torch.as_tensor(space.low, dtype=torch.float32, device=action.device)
    high = torch.as_tensor(space.high, dtype=torch.float32, device=action.device)
    return torch.clamp(action, low, high)


class NormalActionNoise:
  def __init__(self, scale: float = 0.1, warmup_samples: int = 20_000):
    self.scale = scale
    self.warmup_samples = warmup_samples

  def initialize(
    self,
    policy: T.Callable[[core.Observation], core.Action],
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    self._rng = torch.Generator().manual_seed(seed)
    self._action_space = action_space
    self._model_policy = policy
    self._random_policy = RandomPolicy(action_space, rng=self._rng)

  @torch.no_grad()
  def __call__(self, observations: core.Observation, samples: int) -> core.Action:
    if samples <= self.warmup_samples:
      return self._random_policy()
    
    actions = self._model_policy(observations)

    # Compute noises.
    if isinstance(actions, dict):
      return {
        k: self._compute_noise(self._action_space.spaces[k], v)  # type: ignore
        for k, v in actions.items()
      }
    else:
      return self._compute_noise(self._action_space, actions)  # type: ignore

  def record(self, resets: torch.Tensor) -> None:
    pass

  def _compute_noise(self, space: gymnasium.spaces.Box, action: torch.Tensor) -> torch.Tensor:
    noises = self.scale * torch.randn(action.shape, generator=self._rng, device=action.device)
    action = action + noises
    low = torch.as_tensor(space.low, dtype=torch.float32, device=action.device)
    high = torch.as_tensor(space.high, dtype=torch.float32, device=action.device)
    return torch.clamp(action, low, high)


class OrnsteinUhlenbeckActionNoise:
  def __init__(
    self,
    scale: float = 0.1,
    clip: float = 2,
    theta: float = 0.15,
    dt: float = 1e-2,
    warmup_samples: int = 20_000,
  ):
    self.scale = scale
    self.clip = clip
    self.theta = theta
    self.dt = dt
    self.warmup_samples = warmup_samples

  def initialize(
    self,
    policy: T.Callable[[core.Observation], core.Action],
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    self._rng = torch.Generator().manual_seed(seed)
    self._action_space = action_space
    self._model_policy = policy
    self._random_policy = RandomPolicy(action_space, rng=self._rng)
    self._noises: dict[str, torch.Tensor] | None = None

  @torch.no_grad()
  def __call__(self, observations: core.Observation, samples: int) -> core.Action:
    if samples <= self.warmup_samples:
      return self._random_policy()
    
    actions = self._model_policy(observations)

    # Initialize noises.
    if self._noises is None:
      if isinstance(actions, dict):
        self._noises = {k: torch.zeros_like(v) for k, v in actions.items()}
      else:
        self._noises = {'action': torch.zeros_like(actions)}

    # Compute noises.
    if isinstance(actions, dict):
      return {
        k: self._compute_noise(self._action_space.spaces[k], v, k)  # type: ignore
        for k, v in actions.items()
      }
    else:
      return self._compute_noise(self._action_space, actions, 'action')  # type: ignore
  
  def record(self, resets: torch.Tensor) -> None:
    assert self._noises is not None
    for k in self._noises:
      self._noises[k] = self._noises[k] * ~resets[:, None]

  def _compute_noise(self, space: gymnasium.spaces.Box, action: torch.Tensor, k: str) -> torch.Tensor:
    assert self._noises is not None
    noises: torch.Tensor = self._noises[k]
    noise = torch.randn(action.shape, generator=self._rng, device=action.device)
    noise = torch.clamp(noise, -self.clip, self.clip)
    noises = noises - self.theta * noises * self.dt
    noises = noises + self.scale * torch.sqrt(torch.tensor(self.dt)) * noise
    self._noises[k] = noises
    action = action + noises
    low = torch.as_tensor(space.low, dtype=torch.float32, device=action.device)
    high = torch.as_tensor(space.high, dtype=torch.float32, device=action.device)
    return torch.clamp(action, low, high)
