"""
This module defines the `Simulator` class, which manages the computations of a neural circuit tree.
"""
import functools

import torch

from circuits import config, generators, monitors, nodes, units

__all__ = ['Simulator']


def _simulator_init(node: nodes.ModuleNode, cfg: config.SimulatorConfig):
  """Initializes `node` if it is a generator, unit or monitor."""
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.init(cfg)


def _simulator_reset(node: nodes.ModuleNode, reset: torch.Tensor | None = None):
  """Resets `node` if it is a generator, unit or monitor."""
  if isinstance(node, (generators.Generator, units.Unit, monitors.Monitor)):
    node.reset(reset)


def _simulator_step(node: nodes.ModuleNode):
  """Updates the value of `node` if it is a generator or unit (first phase of a step)."""
  # Update values, no dependencies.
  if isinstance(node, (generators.Generator, units.Unit)):
    node.step()


def _simulator_populate(node: nodes.ModuleNode):
  """Reads the new values of `node`'s sources if it is a unit or monitor (second phase of a
  step)."""
  # Read values, via dependencies.
  if isinstance(node, (units.Unit, monitors.Monitor)):
    node.populate()


class Simulator(nodes.ContainerNode):
  """The top-level node that manages the computations of a neural circuit tree. It recursively
  initializes, resets, and updates nodes.

  Each step has two phases: every unit and generator first updates its value from its synapses'
  delayed inputs (`step`), then every unit reads its sources' new values into its synapses
  (`populate`). So the result does not depend on the order of the nodes in the tree, and a synapse
  of delay `d` delivers a value `d` steps after its source produced it.

  Every state tensor has shape `[1]` (or the shape of its signal). To simulate a batch, map the
  simulator over the batch with `torch.func` (e.g. `torch.func.vmap` of `torch.func.functional_call`
  with batched parameters and buffers); `reset` then takes a per-sample bool.
  """
  def __init__(self, **kwargs):
    """Initializes the simulator.

    Args:
      **kwargs: Fields of its `config.SimulatorConfig` (e.g. `timestep`).
    """
    super().__init__()
    self.cfg = config.SimulatorConfig(**kwargs)
    assert self.cfg.timestep > 0

    self.steps = torch.nn.Buffer(torch.zeros(1, dtype=torch.int64), persistent=False)
    """Number of elapsed simulated steps."""

    self.time = torch.nn.Buffer(torch.zeros(1), persistent=False)
    """Duration of elapsed simulated time (in milliseconds), which depends on `cfg.timestep`."""

  def init(self):
    """Initializes the simulator and resets it to its initial state."""
    nodes.visit(self, functools.partial(_simulator_init, cfg=self.cfg))
    self.reset()

  def reset(self, reset: torch.Tensor | None = None):
    """Resets the simulator to its initial state.

    Args:
      reset: Whether to reset, bool `[1]` (per sample when vmapped), or `None` to reset.
    """
    nodes.visit(self, functools.partial(_simulator_reset, reset=reset))
    self.steps[:] = 0 if reset is None else ~reset * self.steps
    self.time[:] = 0 if reset is None else ~reset * self.time

  def step(self):
    """Advances the simulator by one timestep (see the class for the two phases)."""
    # Some nodes may have dependencies on others, so we need to traverse in the right order.
    nodes.visit(self, _simulator_step)
    nodes.visit(self, _simulator_populate)
    self.steps += 1
    self.time += self.cfg.timestep
