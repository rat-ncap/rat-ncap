"""
This module defines `Unit` classes, which model neurons and dynamical motifs.
"""
import abc
import typing as T

import numpy as np
import torch

from circuits import connector, signal
from utils.torch import activations

__all__ = [
  'Unit', 'Basic', 'Adaptor', 'Oscillator',
  'interpolate_linear', 'interpolate_logarithmic', 'clip_normalized',
  'rectangular', 'rectangular_amplitude_modulated', 'linear', 'logarithmic', 'adapting',
]

def _conditional_set(field: torch.Tensor, value: torch.Tensor, condition: torch.Tensor | None = None):
  """Sets `field` to `value` where `condition` is true, otherwise leaves it unchanged."""
  field[:] = value if condition is None else (condition * value + ~condition * field)


def _repr_param(x: T.Any) -> str:
  """Formats a parameter value compactly for `extra_repr` (3 significant digits, tensor shapes)."""
  if isinstance(x, torch.Tensor):
    name = x.__class__.__name__
    if len(x.shape) == 0 and x.device.type != 'meta':
      return f'{name}({x.item():.3g})'
    else:
      return f'{name}{list(x.shape)}'
  elif isinstance(x, float):
    return f'{x:.3g}'
  elif isinstance(x, list):
    return '[' + ', '.join(_repr_param(v) for v in x) + ']'
  elif isinstance(x, tuple):
    return '(' + ', '.join(_repr_param(v) for v in x) + ')'
  else:
    return str(x)


class Unit(signal.Signal, abc.ABC):
  """A primitive node representing an abstract computational unit in a neural circuit."""
  def __init__(self, **kwargs):
    """Initializes the unit with its metadata (`kwargs`)."""
    super().__init__(**kwargs)

  def step(self):
    """Advances the unit to the next state."""
    pass

  def populate(self):
    """Populates the unit with any necessary internal state (e.g. inputs from connections)."""
    pass


class Basic(Unit):
  """A basic neuron with an internal voltage variable. The activation function defines the
  voltage-firing relationship.

  The voltage relaxes toward the input plus the bias, clamped to [-1, +1], with time constant
  `voltage_time / 4`, and the value is `activation(v)`.
  """
  def __init__(
    self,
    activation: T.Callable[[torch.Tensor], torch.Tensor] = activations.unitrelu,
    bias: float | torch.Tensor = 0,
    voltage_time: float = 50,
    **kwargs,
  ):
    """Initializes the neuron.

    Args:
      activation: Activation function of the voltage, `activation(v)`.
      bias: Bias voltage between -1 and +1.
      voltage_time: Time constant of the voltage variable (in milliseconds).
      **kwargs: Metadata of the node.
    """
    super().__init__(**kwargs)
    assert 0 < voltage_time

    self.activation = activation
    self.bias = bias
    self.voltage_time = voltage_time

    self.synapses = connector.Connector(self)

    self.v = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Voltage variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs) -> T.Self:
    """Connects `source` as an input of this unit (see `connector.Connector.connect`), and returns
    the unit, for chaining."""
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg) -> None:
    """Stores the simulator's timestep."""
    self.dt = cfg.timestep

  def reset(self, reset: torch.Tensor | None = None) -> None:
    """Resets the state and the synapses (where `reset` is true, if given)."""
    self.synapses.reset(reset)
    _conditional_set(self.v, torch.zeros(1), reset)
    _conditional_set(self.value, torch.zeros(1), reset)

  def step(self) -> None:
    """Updates the voltage and value from the synapses' input."""
    v = self.v
    y = self.activation

    z = torch.clamp(self.synapses.integrate() + self.bias, min=-1, max=1)

    # Forward-Euler discretization (for reference).
    # dv = z - v
    # self.v[:] = v + 4 * dv / self.voltage_time * self.dt

    # Backward-Euler discretization.
    kv = 4 / self.voltage_time * self.dt
    self.v[:] = (v + z * kv) / (1 + kv)

    self.value[:] = y(self.v)

  def populate(self) -> None:
    """Reads the current values of the input sources into the synapses."""
    self.synapses.populate()

  def extra_repr(self) -> str:
    """Returns the unit's parameters for its `repr`."""
    cfg = dict(
      B=self.bias,
      Tv=self.voltage_time,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])


class Adaptor(Unit):
  """An adapting neuron with an internal voltage and adaptation variable. The `adaptation` constant
  defines if it is depressing (firing less for the same input) or facilitating (firing more for the
  same input). The activation function defines the voltage-firing relationship."""
  def __init__(
    self,
    activation: T.Callable[[torch.Tensor], torch.Tensor] = activations.unitrelu,
    bias: float | torch.Tensor = 0,
    adaptation: float = 0,
    adaptation_time: float = 1000,
    voltage_time: float = 50,
    **kwargs,
  ):
    """Initializes the neuron.

    Args:
      activation: Activation function of the voltage, `activation(v)`.
      bias: Bias voltage between -1 and +1.
      adaptation: Adaptation constant between -1 and +1, representing maximum depression (with
        asymptotic firing rate of zero) and +1 is maximum facilitation (with fastest time to maximum
        firing).
      adaptation_time: Time constant of the adaptation variable (in milliseconds).
      voltage_time: Time constant of the voltage variable (in milliseconds).
      **kwargs: Metadata of the node.
    """
    super().__init__(**kwargs)
    assert -1 <= adaptation <= 1
    assert not (adaptation == 1 and bias == 0)
    assert 0 < adaptation_time
    assert 0 < voltage_time

    self.activation = activation
    self.bias = bias
    self.adaptation = adaptation
    self.adaptation_time = adaptation_time
    self.voltage_time = voltage_time

    self.synapses = connector.Connector(self)

    self.v = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Voltage variable
    self.a = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Adaptation variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs) -> T.Self:
    """Connects `source` as an input of this unit (see `connector.Connector.connect`), and returns
    the unit, for chaining."""
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg) -> None:
    """Stores the simulator's timestep."""
    self.dt = cfg.timestep

  def reset(self, reset: torch.Tensor | None = None) -> None:
    """Resets the state and the synapses (where `reset` is true, if given)."""
    self.synapses.reset(reset)
    _conditional_set(self.v, torch.zeros(1), reset)
    _conditional_set(self.a, torch.zeros(1), reset)
    _conditional_set(self.value, torch.zeros(1), reset)

  def step(self) -> None:
    """Updates the voltage, adaptation and value from the synapses' input."""
    v = self.v
    a = self.a
    y = self.activation

    z = torch.clamp(self.synapses.integrate() + self.bias, min=-1, max=1)

    # Forward-Euler discretization (for reference).
    # dv = z - v + a
    # if self.adaptation > 0:
    #   # Positive adaptation -> facilitation.
    #   da = self.adaptation * y(v) * (1 - torch.clamp(z, 0, 1)) - a
    # else:
    #   # Negative adaptation -> depression.
    #   da = self.adaptation * torch.sign(y(v)) * torch.clamp(z, 0, 1) - a
    # self.v[:] = v + 4 * dv / self.voltage_time * self.dt
    # self.a[:] = a + 4 * da / self.adaptation_time * self.dt

    # Backward-Euler discretization.
    kv = 4 / self.voltage_time * self.dt
    self.v[:] = (v + (z + a) * kv) / (1 + kv)
    if self.adaptation > 0:
      # Positive adaptation -> facilitation.
      x = self.adaptation * y(self.v) * (1 - torch.clamp(z, 0, 1))
    else:
      # Negative adaptation -> depression.
      x = self.adaptation * torch.sign(y(self.v)) * torch.clamp(z, 0, 1)
    ka = 4 / self.adaptation_time * self.dt
    self.a[:] = (a + x * ka) / (1 + ka)

    self.value[:] = y(self.v)

  def populate(self) -> None:
    """Reads the current values of the input sources into the synapses."""
    self.synapses.populate()

  def extra_repr(self) -> str:
    """Returns the unit's parameters for its `repr`."""
    cfg = dict(
      B=self.bias,
      A=self.adaptation,
      Ta=self.adaptation_time,
      Tv=self.voltage_time,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])


def interpolate_linear(
  start: torch.Tensor | float,
  end: torch.Tensor | float,
  z: torch.Tensor | float,
) -> torch.Tensor | float:
  """Interpolate linearly, such that z = 0 maps to `start` and z = 1 maps to `end`."""
  return start * (1 - z) + end * z


def interpolate_logarithmic(
  start: torch.Tensor | float,
  end: torch.Tensor | float,
  z: torch.Tensor | float,
  *,
  base: float = 2,
) -> torch.Tensor | float:
  """Interpolate logarithmically, such that z = 0 maps to `start` and z = 1 maps to `end`."""
  assert base > 0 and base != 1
  b = torch.as_tensor(base)
  return start + (end - start) * torch.log((b - 1) * z + 1) / torch.log(b)


def clip_normalized(
  start: torch.Tensor | float,
  end: torch.Tensor | float,
  z: torch.Tensor | float,
) -> torch.Tensor | float:
  """Normalize linearly, such that z = `start` maps to 0 and z = `end` maps to 1."""
  return torch.clip((z - start) / (end - start), 0, 1)  # type: ignore


# Oscillator activations: `activation(v, a, x, active_bound, quiet_bound, **kwargs)` returns the
# value of an `Oscillator` from its voltage `v` (positive when active), adaptation `a`, input
# `x`, and the adaptation bounds `active_bound` (where the active state ends) and `quiet_bound`
# (where the quiet state ends). Keyword arguments configure the curve (e.g. with
# `functools.partial`).


def rectangular(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active: torch.Tensor | float = 1.,
  quiet: torch.Tensor | float = 0.,
) -> torch.Tensor:
  """Returns `active` when active and `quiet` when quiet."""
  return (v >= 0) * active + (v < 0) * quiet


def rectangular_amplitude_modulated(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active_start: torch.Tensor | float = 0.,
  active_end: torch.Tensor | float = 1.,
  quiet: torch.Tensor | float = 0.,
  base: float = 1.,
) -> torch.Tensor:
  """Returns, when active, `active_start` to `active_end` as the input `x` goes from 0 to 1
  (linearly if `base` is 1, else logarithmically with `base`), and `quiet` when quiet."""
  pos = torch.clip(x, 0, 1)
  if base == 1:
    return (v >= 0) * interpolate_linear(active_start, active_end, pos) + (v < 0) * quiet
  else:
    return ((v >= 0) * interpolate_logarithmic(active_start, active_end, pos, base=base) +
            (v < 0) * quiet)


def linear(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active_start: torch.Tensor | float = 1.0,
  active_end: torch.Tensor | float = 0.5,
  quiet: torch.Tensor | float = 0.,
) -> torch.Tensor:
  """Returns, when active, `active_start` to `active_end` linearly as the adaptation goes from
  `quiet_bound` to `active_bound` over the active state, and `quiet` when quiet."""
  return (
    (v >= 0) *
    interpolate_linear(active_end, active_start, clip_normalized(active_bound, quiet_bound, a)) +
    (v < 0) * quiet
  )


def logarithmic(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active_start: torch.Tensor | float = 1.0,
  active_end: torch.Tensor | float = 0.5,
  quiet: torch.Tensor | float = 0.,
  base: float = 2.,
) -> torch.Tensor:
  """Returns, when active, `active_start` to `active_end` logarithmically (with `base`) as the
  adaptation goes from `quiet_bound` to `active_bound` over the active state, and `quiet` when
  quiet."""
  return ((v >= 0) * interpolate_logarithmic(
    active_end, active_start, clip_normalized(active_bound, quiet_bound, a), base=base
  ) + (v < 0) * quiet)


def adapting(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active_start: torch.Tensor | float = 1.0,
  active_end: torch.Tensor | float = 0.5,
  quiet: torch.Tensor | float = 0.,
) -> torch.Tensor:
  """Returns, when active, `active_end` to `active_start` linearly with the adaptation `a` from 0
  to 1, and `quiet` when quiet."""
  return (v >= 0) * interpolate_linear(active_end, active_start, a) + (v < 0) * quiet


class Oscillator(Unit):
  """A bursting neuron with internal voltage and adaptation variables. The neuron switches its
  voltage variable between active/quiet states based on the adaptation variable. Both excitatory
  (positive) and inhibitory (negative) input scale the active/quiet durations, and possibly the
  adaptation depletion/recovery timescale. Strong positive net input causes (stable) tonic firing,
  while strong negative net input causes (stable) suppression.

  The noise (see `noise`) is drawn from the global torch random number generator, so a simulation is
  reproducible only if the caller seeds it (e.g. `torch.manual_seed`) before simulating.
  """
  def __init__(
    self,
    activation: T.Callable[..., torch.Tensor] = adapting,
    bias: float | torch.Tensor = 0,
    adaptation_time: float = 2000,
    active_time: float = 500,
    quiet_time: float = 1000,
    active_scale_pos: float = 0.5,
    quiet_scale_pos: float = 0.1,
    active_scale_neg: float = 0.7,
    quiet_scale_neg: float = 1.3,
    adaptation_scale_active_pos: float = 1.,
    adaptation_scale_quiet_pos: float = 1.,
    adaptation_scale_active_neg: float = 1.,
    adaptation_scale_quiet_neg: float = 1.,
    active_stable: float = 1.,
    quiet_stable: float = 0.,
    active_stable_margin: float = 0.,
    quiet_stable_margin: float = 0.,
    active_delay: float = 0.,
    quiet_delay: float = 0.,
    noise: float = 0.,
    **kwargs,
  ):
    """Initializes the neuron.

    Args:
      activation: Activation function `activation(v, a, x, active_bound, quiet_bound)` (see
        `adapting`).
      bias: Bias voltage between -1 and +1.
      adaptation_time: Time constant of the adaptation variable (in milliseconds).
      active_time: Duration of the active state (in milliseconds).
      quiet_time: Duration of the quiet state (in milliseconds).
      active_scale_[pos/neg]: Scaling factor of the active state as input goes to [+1/-1].
      quiet_scale_[pos/neg]: Scaling factor of the quiet state as input goes to [+1/-1].
      adaptation_scale_[active/quiet]_[pos/neg]: Scaling factor of the adaptation time constant.
      active_stable: Input strength above which the active state is stable (tonic firing).
      quiet_stable: Input strength below which the quiet state is stable (suppressed).
      active_stable_margin: Time margin for active stable point clipping (in milliseconds).
      quiet_stable_margin: Time margin for quiet stable point clipping (in milliseconds).
      active_delay: Fraction of the active phase time appropriated for the delay.
      quiet_delay: Fraction of the quiet phase time appropriated for the delay.
      noise: Standard deviation of the multiplicative noise on dynamics (mean 1).
      **kwargs: Metadata of the node.
    """
    super().__init__(**kwargs)
    assert 0 < adaptation_time
    assert 0 < active_time
    assert 0 < quiet_time
    assert 0 < active_scale_pos
    assert 0 < quiet_scale_pos
    assert 0 < active_scale_neg
    assert 0 < quiet_scale_neg
    assert 0 < adaptation_scale_active_pos
    assert 0 < adaptation_scale_quiet_pos
    assert 0 < adaptation_scale_active_neg
    assert 0 < adaptation_scale_quiet_neg
    assert quiet_stable <= 0 <= active_stable
    assert 0 <= active_stable_margin
    assert 0 <= quiet_stable_margin
    assert 0 <= active_delay < 1
    assert 0 <= quiet_delay < 1
    assert 0 <= noise

    self.activation = activation
    self.bias = bias
    self.adaptation_time = adaptation_time
    self.active_time = active_time
    self.quiet_time = quiet_time
    self.active_scale_pos = active_scale_pos
    self.quiet_scale_pos = quiet_scale_pos
    self.active_scale_neg = active_scale_neg
    self.quiet_scale_neg = quiet_scale_neg
    self.adaptation_scale_active_pos = adaptation_scale_active_pos
    self.adaptation_scale_quiet_pos = adaptation_scale_quiet_pos
    self.adaptation_scale_active_neg = adaptation_scale_active_neg
    self.adaptation_scale_quiet_neg = adaptation_scale_quiet_neg
    self.active_stable = active_stable
    self.quiet_stable = quiet_stable
    self.active_stable_margin = active_stable_margin
    self.quiet_stable_margin = quiet_stable_margin
    self.active_delay = active_delay
    self.quiet_delay = quiet_delay
    self.noise = noise

    Ta = 4 * active_time / adaptation_time
    Tq = 4 * quiet_time / adaptation_time
    Ta_pos = Ta * active_scale_pos / adaptation_scale_active_pos
    Tq_pos = Tq * quiet_scale_pos / adaptation_scale_quiet_pos
    Ta_neg = Ta * active_scale_neg / adaptation_scale_active_neg
    Tq_neg = Tq * quiet_scale_neg / adaptation_scale_quiet_neg
    self.active_bound_zero = float((1 - np.exp(Tq)) / (1 - np.exp(Ta + Tq)))
    self.quiet_bound_zero = float(self.active_bound_zero * np.exp(Ta))
    self.active_bound_pos = float((1 - np.exp(Tq_pos)) / (1 - np.exp(Ta_pos + Tq_pos)))
    self.quiet_bound_pos = float(self.active_bound_pos * np.exp(Ta_pos))
    self.active_bound_neg = float((1 - np.exp(Tq_neg)) / (1 - np.exp(Ta_neg + Tq_neg)))
    self.quiet_bound_neg = float(self.active_bound_neg * np.exp(Ta_neg))

    self.synapses = connector.Connector(self)

    self.v = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Voltage variable
    self.a = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Adaptation variable
    self.x = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Input variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs) -> T.Self:
    """Connects `source` as an input of this unit (see `connector.Connector.connect`), and returns
    the unit, for chaining."""
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg) -> None:
    """Stores the simulator's timestep."""
    self.dt = cfg.timestep

  def reset(self, reset: torch.Tensor | None = None) -> None:
    """Resets the state and the synapses (where `reset` is true, if given)."""
    self.synapses.reset(reset)
    _conditional_set(self.v, -torch.ones(1), reset)
    _conditional_set(self.a, torch.ones(1) * self.active_bound_zero, reset)
    _conditional_set(self.x, torch.zeros(1), reset)
    _conditional_set(self.value, torch.zeros(1), reset)

  def step(self) -> None:
    """Updates the voltage, adaptation, input and value from the synapses' input."""
    v = self.v
    a = self.a
    y = self.activation
    dt = self.dt

    x = self.synapses.integrate()
    x = x + self.bias
    pos = torch.clamp(x, min=0, max=1)  # Interpolation amounts.
    neg = torch.clamp(-x, min=0, max=1)
    pos_mask = (x >= 0)  # Element-wise masks.
    neg_mask = (x < 0)

    # Interpolate adaptation time constant.
    k = 4 / self.adaptation_time
    k_active = k / (
      interpolate_linear(1, self.adaptation_scale_active_pos, pos) * pos_mask +
      interpolate_linear(1, self.adaptation_scale_active_neg, neg) * neg_mask
    )
    k_quiet = k / (
      interpolate_linear(1, self.adaptation_scale_quiet_pos, pos) * pos_mask +
      interpolate_linear(1, self.adaptation_scale_quiet_neg, neg) * neg_mask
    )

    # Interpolate active/quiet bounds used for state switching.
    active_bound = (
      interpolate_linear(self.active_bound_zero, self.active_bound_pos, pos) * pos_mask +
      interpolate_linear(self.active_bound_zero, self.active_bound_neg, neg) * neg_mask
    )
    quiet_bound = (
      interpolate_linear(self.quiet_bound_zero, self.quiet_bound_pos, pos) * pos_mask +
      interpolate_linear(self.quiet_bound_zero, self.quiet_bound_neg, neg) * neg_mask
    )

    # Calculate multiplicative noise (mean 1, std `self.noise`), always positive.
    noise = torch.clamp(1 + self.noise * torch.randn_like(v), min=0)

    # Adaptation constraints: Clip to bounds ...
    #   (1) before integration, if within stable point time margin --> moves to new stable point
    active_bound_with_margin = active_bound * torch.exp(-self.active_stable_margin * k_active)
    quiet_bound_with_margin = 1 + (quiet_bound - 1) * torch.exp(-self.quiet_stable_margin * k_quiet)
    update_active_stable = (v > 0) & (active_bound_with_margin <= a) & (a <= active_bound)
    update_quiet_stable = (v < 0) & (quiet_bound <= a) & (a <= quiet_bound_with_margin)
    clip = update_active_stable | update_quiet_stable
    a = ~clip * a + clip * torch.clip(a, active_bound, quiet_bound)

    # Adaptation derivative: Moves when within bounds.
    #
    # Forward-Euler discretization (for reference):
    # da = (
    #   if [inside active bound], [decrease a] [at adaptation timescale] [rescaled for delay]
    #   if [inside quiet bound], [increase a] [at adaptation timescale] [rescaled for delay]
    # ) * [speed-up/slow-down with noise]
    # da = (
    #   (v > 0) * (a > active_bound) * (0 - a) * k_active / (1 - self.active_delay) +
    #   (v < 0) * (a < quiet_bound) * (1 - a) * k_quiet  / (1 - self.quiet_delay)
    # ) * noise
    # a_new = a + da * dt
    # a_new = torch.clamp(a_new, min=0, max=1)
    #
    # Backward-Euler discretization:
    # Need to explicitly list all cases unlike `da` above which uses zero when no conditions match.
    t_active = k_active / (1 - self.active_delay) * noise * dt
    t_quiet = k_quiet / (1 - self.quiet_delay) * noise * dt
    inside_active_bound = (v > 0) & (a > active_bound)
    inside_quiet_bound = (v < 0) & (a < quiet_bound)
    otherwise = ~(inside_active_bound | inside_quiet_bound)
    a_new = (
      inside_active_bound * a / (1 + t_active) + \
      inside_quiet_bound * (a + t_quiet) / (1 + t_quiet) + \
      otherwise * a \
    )
    a_new = torch.clip(a_new, 0, 1)

    # Adaptation constraints: Clip to bounds ...
    #   (2) after integration, if just passed the bound on this timestep --> prevents overshoot
    # Clipping ensures the stable point is exactly at the bound elbow and any overshoot should
    # begin a state switch (e.g. for post-inhibitory rebound).
    passed_active_bound = (v > 0) & (a_new <= active_bound) & (a >= active_bound)
    passed_quiet_bound = (v < 0) & (a_new >= quiet_bound) & (a <= quiet_bound)
    clip = passed_active_bound | passed_quiet_bound
    a_new = ~clip * a_new + clip * torch.clip(a_new, active_bound, quiet_bound)
    a = a_new  # Update for use in voltage derivative.

    # Voltage derivative: Moves to elbow when inside bounds, then moves either towards zero for
    # state switch or towards elbow if input strong enough for a stable point (tonic or suppressed).
    # The rate is inversely proportional to the delay time (phase time * delay fraction).
    # The direction is outwards (to +1 or -1) for a stable point, or inwards (to 0) otherwise.
    active_delay = (
      self.active_delay * (torch.log(quiet_bound) - torch.log(active_bound)) / k_active
    )
    quiet_delay = (
      self.quiet_delay * (torch.log(1 - active_bound) - torch.log(1 - quiet_bound)) / k_quiet
    )
    active_rate = 1 / torch.clamp(active_delay, min=1)  # Avoid division by zero.
    quiet_rate = 1 / torch.clamp(quiet_delay, min=1)
    inside_active_bound = (v > 0) & (a > active_bound)
    inside_quiet_bound = (v < 0) & (a < quiet_bound)
    toward_active_stable = (v > 0) & (a == active_bound) & (x > self.active_stable)
    toward_quiet_stable = (v < 0) & (a == quiet_bound) & (x < self.quiet_stable)
    active_direction = torch.sign(+1 * (inside_active_bound | toward_active_stable) - v)
    quiet_direction = torch.sign(-1 * (inside_quiet_bound | toward_quiet_stable) - v)
    # Forward-Euler discretization = Backward-Euler discretization for constant derivative:
    dv = ( \
      (v > 0) * active_rate * active_direction + \
      (v < 0) * quiet_rate * quiet_direction \
    ) * noise
    # Clamp overshoots from large rates (resulting from small/zero delay fractions).
    v_new = torch.clamp(v + dv * dt, min=-1, max=+1)

    # Voltage constraints: Flip if adaptation at/passed bounds and voltage at/passed threshold.
    flip_to_quiet = (a_new <= active_bound) & (v_new <= 0)
    flip_to_active = (a_new >= quiet_bound) & (v_new >= 0)
    v_new = ~flip_to_quiet * v_new + flip_to_quiet * -1
    v_new = ~flip_to_active * v_new + flip_to_active * +1
    v = v_new  # Update after constraint.

    self.v[:] = v
    self.a[:] = a
    self.x[:] = x
    self.value[:] = y(
      v,
      a,
      x,
      active_bound=self.active_bound_pos,
      quiet_bound=self.quiet_bound_zero,
    )

  def populate(self) -> None:
    """Reads the current values of the input sources into the synapses."""
    self.synapses.populate()

  def extra_repr(self) -> str:
    """Returns the unit's parameters for its `repr`."""
    cfg = dict(
      B=self.bias,
      T=self.adaptation_time,
      Tq=self.quiet_time,
      Ta=self.active_time,
      KTq=[self.quiet_scale_neg, self.quiet_scale_pos],
      KTa=[self.active_scale_neg, self.active_scale_pos],
      KT=[
        self.adaptation_scale_quiet_neg,
        self.adaptation_scale_quiet_pos,
        self.adaptation_scale_active_neg,
        self.adaptation_scale_active_pos,
      ],
      D=[self.quiet_delay, self.active_delay],
      S=[self.quiet_stable, self.active_stable],
      N=self.noise,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])
