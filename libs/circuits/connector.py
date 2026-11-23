"""
This module defines `Connector` classes, which model synapses and connectivity motifs.
"""
import dataclasses
import typing as T

import torch

from circuits import signal
from utils import collections
from utils.torch import buffers

__all__ = ['Connection', 'Connector']


@dataclasses.dataclass
class Connection:
  """A connection of a `Connector` to one source signal (see `Connector.connect`).

  Attributes:
    source: Source signal node.
    attr: Attribute of the source that is read.
    weight: Synaptic weight. A `torch.nn.Parameter` is registered on the connector's parent module.
    delay: Buffer of the weighted values not yet delivered.
    noise: Standard deviation of the Gaussian noise added to the weight.
    meta: Metadata of the connection.
    activation: Activation function applied to the weighted value, or `None` for the identity.
  """
  source: signal.Signal
  attr: str
  weight: float | torch.nn.Parameter
  delay: buffers.DelayBuffer
  noise: float
  meta: collections.AttrDict
  activation: T.Callable[[torch.Tensor], torch.Tensor] | None = None


class Connector:
  """A data structure that connects to `Signal` nodes and integrates their current values into
  an output. It can be viewed as a general way to model synapses and connectivity motifs.

  The current implementation models a chemical synapse through a simple weighted sum of connected
  signals with optional noise and delay. A future implementation may support electrical synapses
  and more complex connectivity patterns (e.g. topographical, probabilistic, multicompartmental).

  As this class is not a node in the circuit tree, any `torch.Tensor`/`torch.nn.Parameter` (e.g.
  connection weights) should be managed externally and attached to a `torch.nn.Module`.
  """
  def __init__(self, parent: torch.nn.Module):
    """Initializes the connector, whose weights and buffers are registered on `parent`."""
    self.connections: dict[str, Connection] = {}
    self.parent = parent

  def connect(
    self,
    source: signal.Signal,
    *,
    attr: str = 'value',
    weight: float | torch.nn.Parameter = 1.,
    delay: int = 1,
    noise: float = 0.,
    activation: T.Callable[[torch.Tensor], torch.Tensor] | None = None,
    name: str | None = None,
    **kwargs,
  ) -> T.Self:
    """Connects a signal source to the connector with a given weight, delay, and/or noise.

    Args:
      source: Source signal node.
      attr: Attribute of the source signal to connect. Default: 'value'.
      weight: Synaptic weight, which can be a scalar or a tensor of the same shape as the
        source signal.
      delay: Delay of the connection (in timesteps): a source value is delivered `delay` steps
        after it is populated. At least 1.
      noise: Standard deviation of the Gaussian noise (mean 0) added to the weight at each step,
        drawn from the global torch random number generator.
      activation: Activation function applied to the synapse's output (the weighted source signal),
        without adding delay. Default: identity.
      name: Name of the connection.
      **kwargs: Metadata to associate with the connection.

    Returns:
      The connector, for chaining.
    """
    if not 0 <= noise <= 1: raise ValueError('Expected 0 <= `noise` <= 1')
    if not delay >= 1: raise ValueError('Expected `delay` >= 1')
    x: torch.Tensor = getattr(source, attr)
    connection = Connection(
      source=source,
      attr=attr,
      weight=weight,
      delay=buffers.DelayBuffer(x.shape, delay),
      noise=noise,
      meta=collections.AttrDict(kwargs),
      activation=activation,
    )
    name = name or str(len(self.connections))
    # In functional mode, torch replaces parameters and buffers registered to any `torch.nn.Module`
    # but not non-`Module` data structures like this `Connector`. Therefore, we must register these
    # to the parent `Module` and must retrieve them on usage.
    if isinstance(weight, torch.Tensor):
      self.parent.register_parameter(f'weight_{name}', weight)
    self.parent.register_buffer(f'buffer_{name}', connection.delay.timesteps, persistent=False)
    self.connections[name] = connection
    return self

  def __len__(self) -> int:
    """Returns the number of connections."""
    return len(self.connections)

  def reset(self, reset: torch.Tensor | None = None):
    """Resets the connections' undelivered values to zeros (where `reset` is true, if given)."""
    for name, c in self.connections.items():
      # Retrieve the buffer from the parent `Module`, as in `integrate()`/`populate()`; in functional
      # mode, `c.delay.timesteps` may still reference the buffer of a previous call.
      c.delay.timesteps = getattr(self.parent, f'buffer_{name}')
      c.delay.fill(0 if reset is None else ~reset * c.delay.timesteps)

  def integrate(self) -> torch.Tensor:
    """Returns the sum of the values delivered by all connections at this step."""
    total = torch.zeros(1)
    for name, c in self.connections.items():
      c.delay.timesteps = getattr(self.parent, f'buffer_{name}')
      total = total + c.delay.get()
    return total

  def populate(self):
    """Reads the current values of all connected signals into the connections' delay buffers."""
    for name, c in self.connections.items():
      x = getattr(c.source, c.attr)
      w = getattr(self.parent, f'weight_{name}', c.weight)
      c.delay.timesteps = getattr(self.parent, f'buffer_{name}')
      if c.noise:
        w = w + torch.randn(getattr(w, 'shape', ())) * c.noise
      x = w * x
      if c.activation is not None:
        x = c.activation(x)
      c.delay.step(x)
