import pytest
import torch

import circuits as cc


def simulate(unit: cc.Basic | cc.Oscillator, steps: int) -> tuple[list[float], list[float]]:
  """Simulates `unit` alone, and returns its value and voltage at each step."""
  sim = cc.Simulator(timestep=10)
  sim.unit = unit
  sim.init()
  values, voltages = [], []
  for _ in range(steps):
    sim.step()
    values.append(unit.value.item())
    voltages.append(unit.v.item())
  return values, voltages


def phase_durations(active: list[bool], timestep: float) -> tuple[list[float], list[float]]:
  """Returns the durations of the complete active and quiet phases (not the first and last)."""
  runs: list[tuple[bool, int]] = []
  for state in active:
    if runs and runs[-1][0] == state:
      runs[-1] = (state, runs[-1][1] + 1)
    else:
      runs.append((state, 1))
  runs = runs[1:-1]
  return (
    [n * timestep for state, n in runs if state],
    [n * timestep for state, n in runs if not state],
  )


# ==================================================================================================
# Basic


@pytest.mark.parametrize(
  'bias, expected',
  [(0.5, 0.5), (-0.5, 0.), (3., 1.)],
  ids=['inside', 'negative', 'clamped'],
)
def test_basic_settles_to_its_clamped_input(bias, expected):
  values, _ = simulate(cc.Basic(bias=bias), 100)

  assert values[-1] == pytest.approx(expected, abs=1e-4)


# ==================================================================================================
# Oscillator


@pytest.mark.parametrize(
  'active_time, quiet_time, adaptation_time',
  [(500, 1000, 2000), (300, 600, 1000)],
)
def test_oscillator_phases_last_active_time_and_quiet_time_at_zero_input(
  active_time, quiet_time, adaptation_time
):
  unit = cc.Oscillator(
    active_time=active_time,
    quiet_time=quiet_time,
    adaptation_time=adaptation_time,
  )

  _, voltages = simulate(unit, 1000)

  active_durations, quiet_durations = phase_durations([v > 0 for v in voltages], timestep=10)

  assert len(active_durations) >= 3
  assert all(abs(d - active_time) <= 20 for d in active_durations)
  assert all(abs(d - quiet_time) <= 20 for d in quiet_durations)


def test_half_center_oscillator_alternates():
  sim = cc.Simulator(timestep=10)
  sim.flx = flx = cc.Oscillator(bias=0.2)
  sim.ext = ext = cc.Basic(bias=0.8)
  flx.synapse(ext, weight=-0.1)
  ext.synapse(flx, weight=-0.9)
  sim.init()
  flx_values, ext_values = [], []
  for _ in range(500):
    sim.step()
    flx_values.append(flx.value.item())
    ext_values.append(ext.value.item())

  flx_active = torch.tensor(flx_values) > 0.1
  ext_value = torch.tensor(ext_values)
  onsets = (flx_active[1:] & ~flx_active[:-1]).sum().item()
  assert onsets >= 2
  assert 0.1 < flx_active.float().mean() < 0.9
  assert ext_value[flx_active].mean() < ext_value[~flx_active].mean()  # Inhibited by flx.


def test_oscillator_fires_tonically_above_active_stable():
  _, voltages = simulate(cc.Oscillator(bias=1.5, active_stable=1.), 300)

  assert min(voltages[10:]) == 1.


def test_oscillator_noise_is_reproducible_with_the_same_torch_seed():
  def simulate_noisy() -> list[float]:
    """Simulates a noisy oscillator after seeding torch."""
    torch.manual_seed(0)
    values, _ = simulate(cc.Oscillator(noise=0.1), 300)
    return values

  assert simulate_noisy() == simulate_noisy()


# ==================================================================================================
# Interpolation


def test_interpolations_map_zero_and_one_to_start_and_end():
  z = torch.tensor([0., 1.])

  torch.testing.assert_close(cc.interpolate_linear(2., 4., z), torch.tensor([2., 4.]))
  torch.testing.assert_close(cc.interpolate_logarithmic(2., 4., z), torch.tensor([2., 4.]))


def test_logarithmic_interpolation_rises_faster_than_linear():
  assert cc.interpolate_logarithmic(0., 1., torch.tensor(0.5), base=4) > 0.5


def test_clip_normalized_maps_start_and_end_to_zero_and_one_and_clips():
  normalized = cc.clip_normalized(1., 3., torch.tensor([0., 1., 2., 3., 4.]))

  torch.testing.assert_close(normalized, torch.tensor([0., 0., 0.5, 1., 1.]))
