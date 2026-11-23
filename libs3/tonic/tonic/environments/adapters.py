import collections
import functools
import os
import typing as T

import gymnasium
import gymnasium.spaces
import numpy as np
import dm_env
import dm_env.specs
import torch

from tonic import core, logging

logger = logging.getLogger()


@functools.singledispatch
def spec_to_space(spec: dm_env.specs.Array | tuple | list | dict) -> gymnasium.spaces.Space:
  """Converts a dm_env spec to a Gymnasium space.
  - specs.Array -> spaces.Box
  - specs.BoundedArray -> spaces.Box
  - specs.DiscreteArray -> spaces.Discrete
  - tuple, list -> spaces.Tuple
  - dict -> spaces.Dict
  """
  raise NotImplementedError(f"Unknown spec: {spec}")


@spec_to_space.register(dm_env.specs.Array)
def spec_to_space_array(spec: dm_env.specs.Array) -> gymnasium.spaces.Box:
  return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=spec.shape, dtype=spec.dtype)  # type: ignore


@spec_to_space.register(dm_env.specs.BoundedArray)
def spec_to_space_bounded_array(spec: dm_env.specs.BoundedArray) -> gymnasium.spaces.Box:
  return gymnasium.spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype)  # type: ignore


@spec_to_space.register(dm_env.specs.DiscreteArray)
def spec_to_space_discrete_array(spec: dm_env.specs.DiscreteArray) -> gymnasium.spaces.Discrete:
  return gymnasium.spaces.Discrete(n=spec.num_values)


@spec_to_space.register(tuple)
@spec_to_space.register(list)
def spec_to_space_tuple(spec: tuple | list) -> gymnasium.spaces.Tuple:
  return gymnasium.spaces.Tuple(tuple(spec_to_space(subspec) for subspec in spec))


@spec_to_space.register(dict)
def spec_to_space_dict(spec: dict) -> gymnasium.spaces.Dict:
  # Need `OrderedDict` to prevent sorting upon initialization.
  return gymnasium.spaces.Dict(
    collections.OrderedDict([(key, spec_to_space(subspec)) for key, subspec in spec.items()]),
  )


class DMEnv(core.Env):
  """Converts a DMEnv environment into a Gymnasium environment."""

  def __init__(self, environment: dm_env.Environment, random: np.random.RandomState):
    self.environment = environment
    self.random = random

    # Convert from `dm_env.specs` to `gymnasium.spaces`.
    self.observation_space = spec_to_space(self.environment.observation_spec())
    self.action_space = spec_to_space(self.environment.action_spec())

  def reset(self, seed: int | None = None) -> tuple[core.Observation, dict]:
    if seed is not None:
      self.random.seed(seed)
      self.observation_space.seed(seed)
      self.action_space.seed(seed)
    timestep = self.environment.reset()
    self.last_timestep = timestep
    return torch.from_numpy(timestep.observation), {}

  def step(self, action: core.Action) -> tuple[core.Observation, float, bool, bool, dict]:
    assert isinstance(action, torch.Tensor)
    try:
      timestep = self.environment.step(action.numpy())
      observation = torch.from_numpy(timestep.observation)
      reward = timestep.reward
      discount = timestep.discount
      # Keep only true terminations, not truncations (i.e. timeouts).
      terminated = timestep.last() and discount == 0.
      truncated = timestep.last() and discount != 0
      self.last_timestep = timestep

    # In case MuJoCo crashed.
    except Exception as e:
      logger.file().save('crashes.txt', f'{e}\n')
      logger.error(str(e))
      observation = torch.from_numpy(self.last_timestep.observation)
      reward = 0.
      terminated = True
      truncated = False

    return observation, reward, terminated, truncated, {}
