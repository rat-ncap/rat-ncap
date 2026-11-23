import typing as T

import torch

import circuits as cc


class Pair(T.NamedTuple):
  """A simulator of two mutually connected units with an input signal (see `make_pair`)."""
  sim: cc.Simulator
  input: cc.Signal
  a: cc.Basic
  b: cc.Basic


def make_pair(order: str) -> Pair:
  """Two mutually connected units `a` and `b` with an input, registered in `order` ('ab'/'ba')."""
  sim = cc.Simulator(timestep=10)
  sim.input = input = cc.Signal()
  a, b = cc.Basic(bias=0.2), cc.Basic(bias=0.5)
  for name in order:
    setattr(sim, name, {'a': a, 'b': b}[name])
  a.synapse(input).synapse(b, weight=-0.8)
  b.synapse(a, weight=0.6, delay=2)
  return Pair(sim, input, a, b)


def run(pair: Pair, steps: int) -> None:
  """Steps the simulator with a sine input."""
  for t in range(steps):
    pair.input.value[:] = torch.sin(torch.tensor(t / 5.))
    pair.sim.step()


def state(sim: cc.Simulator) -> dict[str, torch.Tensor]:
  """Copies the simulator's and units' buffers (not the input signal's, which the caller sets)."""
  return {k: v.clone() for k, v in sim.named_buffers() if not k.startswith('input.')}


class Stepper(torch.nn.Module):
  """Steps or resets a simulator in `forward`, for `torch.func.functional_call`, and returns the
  value of `unit` (which is not registered again, so its parameters have one name each)."""

  def __init__(self, sim: cc.Simulator, unit: cc.Basic):
    super().__init__()
    self.sim = sim
    self.read = lambda: unit.value.clone()

  def forward(self, reset: torch.Tensor | None = None) -> torch.Tensor:
    if reset is not None:
      self.sim.reset(reset)
    else:
      self.sim.step()
    return self.read()


def make_weighted() -> tuple[cc.Simulator, cc.Basic]:
  """A simulator of a signal into a unit through a learnable weight of 1 (`unit.weight_0`)."""
  sim = cc.Simulator(timestep=10)
  sim.input = input = cc.Signal(1.)
  sim.unit = unit = cc.Basic().synapse(input, weight=torch.nn.Parameter(torch.tensor(1.)))
  sim.init()
  return sim, unit


def given_buffers(sim: cc.Simulator) -> dict[str, torch.Tensor]:
  """Copies of the simulator's buffers, named as in `Stepper`."""
  return {f'sim.{k}': v.clone() for k, v in sim.named_buffers()}


# ==================================================================================================
# Simulator


def test_step_result_does_not_depend_on_registration_order():
  ab, ba = make_pair('ab'), make_pair('ba')
  ab.sim.init()
  ba.sim.init()

  run(ab, 30)
  run(ba, 30)

  torch.testing.assert_close(ab.a.value, ba.a.value, rtol=0, atol=0)
  torch.testing.assert_close(ab.b.value, ba.b.value, rtol=0, atol=0)


def test_step_advances_steps_and_time():
  pair = make_pair('ab')
  pair.sim.init()

  run(pair, 3)

  assert pair.sim.steps.item() == 3
  assert pair.sim.time.item() == 30


def test_reset_restores_the_initial_state_only_where_reset_is_true():
  pair = make_pair('ab')
  pair.sim.init()
  initial = state(pair.sim)
  run(pair, 10)
  stepped = state(pair.sim)

  pair.sim.reset(torch.tensor([False]))
  kept = state(pair.sim)
  pair.sim.reset(torch.tensor([True]))

  for name, value in state(pair.sim).items():
    torch.testing.assert_close(kept[name], stepped[name], rtol=0, atol=0)
    torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_functional_call_uses_given_parameters_and_updates_given_buffers():
  sim, unit = make_weighted()
  params = {'sim.unit.weight_0': torch.tensor(0.5)}
  buffers = given_buffers(sim)

  for _ in range(5):
    value = torch.func.functional_call(Stepper(sim, unit), (params, buffers))

  for _ in range(5):
    Stepper(sim, unit)()
  assert 0 < value.item() < unit.value.item()  # Weight 0.5 vs the module's own weight 1.
  assert buffers['sim.steps'].item() == 5  # Given buffers are updated in place.
  torch.testing.assert_close(dict(unit.named_parameters())['weight_0'].detach(), torch.tensor(1.))


def test_functional_reset_resets_the_given_delay_buffers():
  sim, unit = make_weighted()
  first = given_buffers(sim)
  torch.func.functional_call(Stepper(sim, unit), ({}, first))
  second = {k: v.clone() for k, v in first.items()}
  assert second['sim.unit.buffer_0'].abs().sum() > 0

  torch.func.functional_call(Stepper(sim, unit), ({}, second), args=(torch.tensor([True]),))

  assert second['sim.unit.buffer_0'].abs().sum() == 0
  assert first['sim.unit.buffer_0'].abs().sum() > 0  # The previous call's buffers are untouched.
