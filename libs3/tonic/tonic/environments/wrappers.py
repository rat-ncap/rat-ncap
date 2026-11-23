"""Environment wrappers."""

import typing as T

import gymnasium
import gymnasium.spaces
import gymnasium.spaces.utils
import numpy as np
import torch

from tonic.torch import space


class ActionRescaler(gymnasium.ActionWrapper):
  """Rescales actions from [-1, 1]^n to the true action space. The baseline agents return actions 
  in [-1, 1]^n."""

  def __init__(self, env: gymnasium.Env):
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    super().__init__(env)
    high = torch.ones(env.action_space.shape)
    self.action_space = gymnasium.spaces.Box(low=-high, high=high)
    true_low = torch.from_numpy(env.action_space.low)
    true_high = torch.from_numpy(env.action_space.high)
    assert torch.all(torch.isfinite(true_low))
    assert torch.all(torch.isfinite(true_high))
    self.bias = (true_high + true_low) / 2
    self.scale = (true_high - true_low) / 2

  def action(self, action: torch.Tensor) -> torch.Tensor:
    return self.bias + self.scale * torch.clip(action, -1, 1)


class TimeFeature(gymnasium.Wrapper):
  """Adds a notion of time in the observations. It can be used in terminal timeout settings to get 
  Markovian MDPs. """

  def __init__(self, env: gymnasium.Env, max_steps: int, low: float = -1, high: float = +1):
    super().__init__(env)
    assert isinstance(self.observation_space, gymnasium.spaces.Box)
    dtype = self.observation_space.dtype
    self.observation_space = gymnasium.spaces.Box(
      low=np.append(self.observation_space.low, low).astype(dtype),
      high=np.append(self.observation_space.high, high).astype(dtype),
    )
    self.max_episode_steps = max_steps
    self.steps = 0
    self.low = low
    self.high = high

  def reset(self, **kwargs: T.Any) -> torch.Tensor:
    self.steps = 0
    observation, _ = self.env.reset(**kwargs)
    observation = torch.from_numpy(observation) if isinstance(observation, np.ndarray) else observation
    observation = torch.cat([observation, torch.tensor([self.low], dtype=observation.dtype)])
    return observation

  def step(self, action: torch.Tensor) -> T.Tuple[torch.Tensor, T.SupportsFloat, bool, bool, dict]:
    assert self.steps < self.max_episode_steps
    observation, reward, terminated, truncated, info = self.env.step(action)
    self.steps += 1
    prop = self.steps / self.max_episode_steps
    v = self.low + (self.high - self.low) * prop
    observation = torch.from_numpy(observation) if isinstance(observation, np.ndarray) else observation
    observation = torch.cat([observation, torch.tensor([v], dtype=observation.dtype)])
    return observation, reward, terminated, truncated, info


class FlattenObservationAction(gymnasium.Wrapper):
  """Flattens both the `observation_space` and the `action_space`."""

  def __init__(self, env: gymnasium.Env):
    super().__init__(env)
    self.observation_space = gymnasium.spaces.utils.flatten_space(self.env.observation_space)
    self.action_space = gymnasium.spaces.utils.flatten_space(self.env.action_space)

    # Monkey-patch the spaces with pointer to originals. Useful for model modules that need to
    # work with unflattened observations/actions.
    self.observation_space.original = self.env.observation_space  # type: ignore
    self.action_space.original = self.env.action_space  # type: ignore

  def reset(self, **kwargs: T.Any) -> torch.Tensor:
    observation, _ = self.env.reset(**kwargs)
    observation = space.pack_tensors(self.env.observation_space, observation)
    return observation

  def step(self, action: torch.Tensor) -> T.Tuple[torch.Tensor, T.SupportsFloat, bool, bool, dict]:
    action = space.unpack_tensors(self.env.action_space, action)
    observation, reward, terminated, truncated, info = self.env.step(action)
    observation = space.pack_tensors(self.env.observation_space, observation)
    return observation, reward, terminated, truncated, info
