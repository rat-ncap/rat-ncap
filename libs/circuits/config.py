"""
This module defines `SimulatorConfig`, which holds configuration settings for the simulator.
"""
import dataclasses

__all__ = ['SimulatorConfig']


@dataclasses.dataclass
class SimulatorConfig:
  """Configuration settings of a `Simulator`."""
  timestep: float = 10
  """Time (in milliseconds) represented by a single step of the simulation."""
