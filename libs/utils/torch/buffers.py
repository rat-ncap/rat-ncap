import torch


class DelayBuffer:
  """A buffer that stores tensors for a fixed number of timesteps."""
  def __init__(self, shape: tuple[int, ...], delay: int = 1):
    if not (delay >= 1): raise ValueError('Expected delay >= 1')
    self.delay = delay
    self.shape = shape
    self.timesteps = torch.zeros((delay, *shape))

  def clear(self):
    self.timesteps[:] = 0.

  def fill(self, value: torch.Tensor | float = 0.):
    self.timesteps[:] = value

  def step(self, elem: torch.Tensor | float = 0.):
    # Use cat rather than roll to avoid in-place operation.
    elem = torch.as_tensor(elem, dtype=self.timesteps.dtype)
    self.timesteps[:] = torch.cat((elem.unsqueeze(0), self.timesteps[:-1]))

  def get(self, t: int = -1):
    assert -self.delay <= t <= -1, f'Expected -{self.delay} <= t <= -1, got t={t}'
    return self.timesteps[t]

  def __getitem__(self, t: int = -1):
    assert -self.delay <= t <= -1, f'Expected -{self.delay} <= t <= -1, got t={t}'
    return self.timesteps[t]

  def __len__(self):
    return len(self.timesteps)

  def __iter__(self):
    return self.timesteps.__iter__()
