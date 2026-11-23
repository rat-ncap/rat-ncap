"""This module provides optimizers for updating parameters based on estimated gradients."""
import abc
import typing as T

import numpy as np
import torch


class Optimizer(abc.ABC):
  """An optimizer turns estimated gradients into parameter updates, for gradient ascent."""
  def initialize(self, params: torch.Tensor) -> None:
    """Initialize the optimizer with the parameters `params`."""
    pass

  @abc.abstractmethod
  def step(self, grad: torch.Tensor) -> torch.Tensor:
    """Returns the update of the parameters (added to them) for the gradient `grad`."""
    raise NotImplementedError()

  def get_state(self) -> dict[str, T.Any]:
    """Returns the state the optimizer needs to continue (e.g. moment estimates), to save it."""
    return {}

  def set_state(self, state: dict[str, T.Any]) -> None:
    """Restores a state returned by `get_state`."""
    pass


class SGD(Optimizer):
  """Stochastic Gradient Descent (SGD) optimizer: the update is `learning_rate * grad`."""
  def __init__(self, learning_rate: float = 0.1):
    """Initializes the optimizer with its learning rate."""
    self.learning_rate = learning_rate

  @T.override
  def step(self, grad: torch.Tensor) -> torch.Tensor:
    return self.learning_rate * grad


class MomentumSGD(Optimizer):
  """Stochastic Gradient Descent (SGD) optimizer with momentum: the update is
  `v = learning_rate * grad + momentum * v`."""
  def __init__(self, learning_rate: float = 0.1, momentum: float = 0.):
    """Initializes the optimizer with its learning rate and momentum factor."""
    self.learning_rate = learning_rate
    self.momentum = momentum

  @T.override
  def initialize(self, params: torch.Tensor) -> None:
    self.v = torch.zeros_like(params)

  @T.override
  def step(self, grad: torch.Tensor) -> torch.Tensor:
    self.v = self.learning_rate * grad + self.momentum * self.v
    return self.v

  @T.override
  def get_state(self) -> dict[str, T.Any]:
    return {'v': self.v.cpu()}

  @T.override
  def set_state(self, state: dict[str, T.Any]) -> None:
    self.v = state['v'].to(self.v)


class Adam(Optimizer):
  """Adaptive Moment Estimation (Adam) optimizer."""
  def __init__(
    self,
    learning_rate: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-08,
  ):
    """Initializes the optimizer.

    Args:
      learning_rate: Learning rate.
      beta1: Decay rate of the first moment estimate.
      beta2: Decay rate of the second moment estimate.
      epsilon: Constant added to the second moment's square root for numerical stability.
    """
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  @T.override
  def initialize(self, params: torch.Tensor) -> None:
    self.m = torch.zeros_like(params)
    self.v = torch.zeros_like(params)
    self.t = 0

  @T.override
  def step(self, grad: torch.Tensor) -> torch.Tensor:
    self.t += 1
    a = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
    return a * self.m / (torch.sqrt(self.v) + self.epsilon)

  @T.override
  def get_state(self) -> dict[str, T.Any]:
    return {'m': self.m.cpu(), 'v': self.v.cpu(), 't': self.t}

  @T.override
  def set_state(self, state: dict[str, T.Any]) -> None:
    self.m = state['m'].to(self.m)
    self.v = state['v'].to(self.v)
    self.t = state['t']
