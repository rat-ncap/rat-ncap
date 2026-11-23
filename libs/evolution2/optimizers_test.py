import pytest
import torch

from evolution2 import optimizers

GRAD = torch.tensor([[2.0, -0.5, 0.0]])


def test_sgd_update_is_learning_rate_times_gradient():
  optimizer = optimizers.SGD(learning_rate=0.1)
  optimizer.initialize(torch.zeros_like(GRAD))

  torch.testing.assert_close(optimizer.step(GRAD), 0.1 * GRAD)


def test_momentum_sgd_accumulates_past_updates():
  optimizer = optimizers.MomentumSGD(learning_rate=0.1, momentum=0.5)
  optimizer.initialize(torch.zeros_like(GRAD))

  optimizer.step(GRAD)
  update = optimizer.step(-GRAD)

  torch.testing.assert_close(update, 0.1 * -GRAD + 0.5 * 0.1 * GRAD)


def test_adam_first_update_is_learning_rate_times_gradient_sign():
  optimizer = optimizers.Adam(learning_rate=0.1)
  optimizer.initialize(torch.zeros_like(GRAD))

  update = optimizer.step(GRAD)

  torch.testing.assert_close(update, 0.1 * torch.sign(GRAD), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
  'make_optimizer',
  [
    lambda: optimizers.SGD(),
    lambda: optimizers.MomentumSGD(momentum=0.9),
    lambda: optimizers.Adam(),
  ],
  ids=['sgd', 'momentum_sgd', 'adam'],
)
def test_restored_state_reproduces_the_next_update(make_optimizer):
  optimizer = make_optimizer()
  optimizer.initialize(torch.zeros_like(GRAD))
  optimizer.step(GRAD)
  optimizer.step(2 * GRAD)
  restored = make_optimizer()
  restored.initialize(torch.zeros_like(GRAD))

  restored.set_state(optimizer.get_state())

  torch.testing.assert_close(restored.step(-GRAD), optimizer.step(-GRAD), rtol=0, atol=0)
