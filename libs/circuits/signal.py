"""
This defines the `Signal` class, which is a primitive node that represents a tensor-valued
signal.
"""
import torch

from circuits import nodes

__all__ = ['Signal']

class Signal(nodes.PrimitiveNode):
  """A primitive node that represents a tensor-valued signal. It can be connected to other nodes
  that consume signals through the common `value` attribute. Many nodes subclass this and implement
  distinct internal logic, but it is useful to directly instantiate this class for input from code
  external to the circuits library (e.g. writing environment observations).

  Attributes:
    value (torch.Tensor): Current tensor value of the signal, float32, at least 1-D (a scalar is
      stored as shape `[1]`).
  """
  def __init__(self, value: torch.Tensor | float = 0., **kwargs):
    """Initializes the signal.

    Args:
      value: Initial value. A float32 tensor is stored as is (shared, not copied); other values are
        converted to a new float32 tensor.
      **kwargs: Metadata of the node.
    """
    super().__init__(**kwargs)
    # Allows pointer to the same tensor.
    tensor = torch.as_tensor(value, dtype=torch.float)
    if tensor.ndim == 0: tensor = tensor.unsqueeze(0)
    self.value = torch.nn.Buffer(tensor, persistent=False)
    """Current tensor value of the signal."""
