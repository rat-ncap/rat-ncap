import abc
import os
import random

import numpy as np
import torch

from tonic import core


class Agent(abc.ABC):
  model: torch.nn.Module

  def initialize(
    self,
    observation_space: core.ObservationSpace,
    action_space: core.ActionSpace,
    seed: int,
  ) -> None:
    self.observation_space = observation_space
    self.action_space = action_space
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

  @abc.abstractmethod
  def step(self, observations: core.Observation) -> core.Action:
    """Returns actions during training."""
    pass

  @abc.abstractmethod
  def test_step(self, observations: core.Observation) -> core.Action:
    """Returns actions during testing."""
    pass

  def record(
    self,
    observations: core.Observation,
    actions: core.Action,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    next_observations: core.Observation,
  ) -> None:
    """Informs the agent of the latest transitions during training."""
    pass

  def update(self) -> None:
    """Updates the parameters of the agent during training."""
    pass

  def save(self, path: str) -> None:
    """Saves the agent weights to a checkpoint."""
    path = path + '.pt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(self.model.state_dict(), path)

  def load(self, path: str) -> None:
    """Loads the agent weights from a checkpoint."""
    path = path + '.pt'
    self.model.load_state_dict(torch.load(path, map_location=torch.get_default_device()))
