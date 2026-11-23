"""This file ports the original modular Quadruped NCAP architecture of
[ncap-quadruped](https://github.com/nikhilxb/ncap-quadruped) (`ncap/quadruped/models/modular.py`
with `configs/quadruped/model/es/unitree_a1/spinal_bccmd.yaml`), built for the Unitree A1, to the
rat body. The architecture is composed of:

- 1 x **Rhythm Generation** (RG) circuit with the original Oscillator dynamics and wiring.
- 4 x **Pattern Formation** (PF) linear tanh heads driven only by the RG (no observations), shared
  between contralateral limbs.
- 4 x **Afferent Feedback** (AF) linear heads from rectified per-limb observations to the RG, shared
  between contralateral limbs.
- 1 x **Brainstem Command** (BC) signal written into every RG command unit.

Adaptations to the rat: the PF drives the rat's limb actions (e.g. position setpoints through a
`Hardtanh`, or flexor-extensor muscles), and the BC is a fixed affine function of the task's
`target_vx` observation instead of a constant.
"""
import collections
import typing as T

import gymnasium
import numpy as np
import torch

import tonic.torch.models as ttm

import circuits as cc
import circuits.connector as connector
import circuits.units as ccu
import utils.torch as utt

import models.mlp as mlp
import models.ncap as ncap

__all__ = [
  'quadruped_es',
  'LegacyOscillator',
  'init_rg_quadruped',
  'QuadrupedRhythmGenerationCircuit',
  'QuadrupedPatternFormationNetwork',
  'QuadrupedAfferentFeedbackNetwork',
  'QuadrupedBrainstemCommand',
  'QuadrupedNCAPActor',
]

LimbName = T.Literal['FL', 'FR', 'HL', 'HR']
LIMBS: tuple[LimbName, ...] = ('FL', 'FR', 'HL', 'HR')


def quadruped_es(
  pattern_formation: 'QuadrupedPatternFormationNetwork',
  rhythm_generation: 'QuadrupedRhythmGenerationCircuit',
  afferent_feedback: 'QuadrupedAfferentFeedbackNetwork | None' = None,
  brainstem_command: 'QuadrupedBrainstemCommand | None' = None,
  constrainer: 'ncap.Constrainer | None' = None,
  timestep: float = 30.,
) -> ttm.ActorOnly:
  """Builds the original Quadruped NCAP model for evolution strategies: an actor without a critic.

  Args:
    pattern_formation: PF network.
    rhythm_generation: RG circuit.
    afferent_feedback: AF network, or `None` for no afferent feedback.
    brainstem_command: BC module, or `None` for no command.
    constrainer: Constrainer of the parameters. Defaults to a `Constrainer`, which enforces the PF
      and AF weight constraints.
    timestep: Duration of one step of the RG simulation (ms). Must equal the task's control
      timestep (e.g. 30 ms for 0.03 s).

  Returns:
    The model.
  """
  # tonic's `ActorLike` protocol describes its own actors; this actor has the same interface.
  actor = T.cast(ttm.ActorLike, QuadrupedNCAPActor(
    pattern_formation=pattern_formation,
    rhythm_generation=rhythm_generation,
    afferent_feedback=afferent_feedback,
    brainstem_command=brainstem_command,
    constrainer=constrainer or ncap.Constrainer(),
    timestep=timestep,
  ))
  return ttm.ActorOnly(actor=actor)


# ==================================================================================================
# Rhythm Generation module


def halflinear(v: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
  """The original Oscillator activation: `(a + 1) / 2` when active (`v > 0`), else 0. The input
  `z` is unused."""
  return (v > 0) * (a + 1) / 2


class LegacyOscillator(ccu.Unit):
  """The bursting neuron of the original `circuits.Oscillator` (commit 5da9cc7): it switches its
  voltage between active (+1) and quiet (-1) states based on the adaptation variable, excitatory
  input scales the active/quiet durations, and input above `tonic_threshold` fires tonically.
  Rewritten without in-place boolean indexing so that it runs under `torch.vmap`.

  `reset` draws the initial adaptation from the global torch random number generator, so the
  initial state depends on the torch seed (and on `torch.vmap`'s randomness mode).
  """
  def __init__(
    self,
    activation: T.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = halflinear,
    bias: float | torch.Tensor = 0,
    adaptation_time: float = 2000,
    active_time: float = 500,
    quiet_time: float = 1000,
    active_scale: float = 0.5,
    quiet_scale: float = 0.1,
    tonic_threshold: float = 1.0,
    **kwargs,
  ):
    """Initializes the neuron.

    Args:
      activation: Activation `activation(v, a, z)` of the voltage `v`, adaptation `a` and clamped
        input `z`.
      bias: Bias of the input.
      adaptation_time: Time constant of the adaptation variable (ms).
      active_time: Duration of the active state at zero input (ms).
      quiet_time: Duration of the quiet state at zero input (ms).
      active_scale: Factor of the active duration at input 1.
      quiet_scale: Factor of the quiet duration at input 1.
      tonic_threshold: Input above which the neuron stays active (tonic firing).
      **kwargs: Metadata of the node.
    """
    super().__init__(**kwargs)
    assert 0 < adaptation_time
    assert 0 < active_time
    assert 0 < quiet_time
    assert 0 < active_scale
    assert 0 < quiet_scale

    self.activation = activation
    self.bias = bias
    self.adaptation_time = adaptation_time
    self.active_time = active_time
    self.quiet_time = quiet_time
    self.active_scale = active_scale
    self.quiet_scale = quiet_scale
    self.tonic_threshold = tonic_threshold

    ka = 4 * active_time / adaptation_time
    kq = 4 * quiet_time / adaptation_time
    self.active_bound0 = float((1 - np.exp(kq)) / (1 - np.exp(ka + kq)))
    self.quiet_bound0 = float(self.active_bound0 * np.exp(ka))
    self.active_bound1 = float(
      (1 - np.exp(kq * quiet_scale)) / (1 - np.exp(ka * active_scale + kq * quiet_scale))
    )
    self.quiet_bound1 = float(self.active_bound1 * np.exp(ka * active_scale))

    self.synapses = connector.Connector(self)

    self.v = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Voltage variable
    self.a = torch.nn.Buffer(torch.zeros(1), persistent=False)  # Adaptation variable
    self.dt: float  # Timestep

  def synapse(self, source: cc.Signal, **kwargs) -> T.Self:
    """Connects `source` as an input (see `circuits.Connector.connect`), and returns the neuron."""
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg) -> None:
    """Stores the simulator's timestep."""
    self.dt = cfg.timestep

  def reset(self, reset: torch.Tensor | None = None) -> None:
    """Resets to the quiet state with a random adaptation near the active bound (where `reset` is
    true, if given)."""
    self.synapses.reset(reset)
    a0 = torch.rand(1) * (self.quiet_bound0 - self.active_bound0) * 0.1 + self.active_bound0
    ccu._conditional_set(self.v, -torch.ones(1), reset)
    ccu._conditional_set(self.a, a0, reset)
    ccu._conditional_set(self.value, torch.zeros(1), reset)

  def step(self) -> None:
    """Updates the adaptation, voltage and value from the synapses' input."""
    v = self.v
    a = self.a

    x = self.synapses.integrate() + self.bias
    z = torch.clamp(x, min=0, max=1)

    # Backward-Euler discretization.
    k = 4 / self.adaptation_time * self.dt
    a = (v < 0) * (a + k) / (1 + k) + (v > 0) * a / (1 + k)
    a = torch.clamp(a, min=0, max=1)

    active_bound = ccu.interpolate_linear(self.active_bound0, self.active_bound1, z)
    quiet_bound = ccu.interpolate_linear(self.quiet_bound0, self.quiet_bound1, z)
    active_to_quiet = (a <= active_bound) & (x <= self.tonic_threshold)
    quiet_to_active = (a >= quiet_bound) & (x >= 0)
    v = torch.where(active_to_quiet, -1., v)
    v = torch.where(quiet_to_active, +1., v)

    self.v[:] = v
    self.a[:] = a
    self.value[:] = self.activation(v, a, z)

  def populate(self) -> None:
    """Reads the current values of the input sources into the synapses."""
    self.synapses.populate()


def init_rg_quadruped(
  p: utt.containers.ParameterManager,
  grad: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
  """The original RG initialization (`init_rg_default`): configures the unit biases and connection
  weights in `p`, with the parameter names of `ncap.init_rg_rat`.

  Args:
    p: Parameter manager to configure.
    grad: Whether the parameters are trained.

  Returns:
    `(oscillator kwargs, basic unit kwargs)`, the hyperparameters of the `LegacyOscillator`
    flexors and extensors and of the `Basic` interneurons.
  """
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], 0.1, grad)
  p.config('bias_ext', ['bias_ext_*'], 0.8, grad)
  p.config('bias_v0d', ['bias_v0d_*'], -0.1, grad)
  p.config('bias_v0v', ['bias_v0v_*'], -0.1, grad)
  p.config('bias_v3f', ['bias_v3f_*'], -0.4, grad)
  p.config('bias_v3e', ['bias_v3e_*'], -0.0, grad)
  p.config('bias_v3a', ['bias_v3a_*'], -0.4, grad)
  p.config('bias_in2', ['bias_in2_*'], -0.1, grad)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -1.0, grad)
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.1, grad)
  # Cross-side connections (cross).
  p.config('cross_flxflx_i', ['cross_flxflx_i_*'], -1.5, grad)
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.3, grad)
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.05, grad)
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -0.1, grad)
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.1, grad)
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.1, grad)
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -0.8, grad)
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.2, grad)
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.2, grad)
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.2, grad)
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.8, grad)
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1.0, grad)
  p.config('aff_ext', ['aff_ext_*'], 1.0, grad)
  # Command connections (freq, sync).
  p.config('freq_osc_e', ['freq_osc_e_*'], 1.0, grad)
  p.config('freq_cross_e', ['freq_cross_e_*'], 1.0, grad)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.8, grad)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.8, grad)
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -0.5, grad)
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.5, grad)
  # Unit parameters.
  osc = {
    'adaptation_time': 1400,
    'active_time': 200,
    'quiet_time': 1200,
    'active_scale': 100 / 200,
    'quiet_scale': 60 / 1200,
    'tonic_threshold': 3,
  }
  basic = {
    'voltage_time': 30,
  }
  return osc, basic


class QuadrupedRhythmGenerationCircuit(cc.Group):
  """The original Rhythm Generation (RG) circuit (Danner et al. 2019) of the Quadruped NCAP, with
  the original units, connections, and connection noise. The limbs are denoted "FL" (front left),
  "FR" (front right), "HL" (hind left), and "HR" (hind right).

  The units and parameters are named as `ncap.RhythmGenerationCircuit`'s, and the outputs are each
  limb's `[flx, ext]`. The layout constants and the `xy`, `flow`, `anchor` and `waypoints`
  arguments only place the units in circuit diagrams.
  """
  def __init__(self, init: T.Callable = init_rg_quadruped, **kwargs):
    """Initializes the circuit.

    Args:
      init: Initializer of the parameters and unit hyperparameters (see `init_rg_quadruped`).
      **kwargs: Metadata of the group node (e.g. for rendering).
    """
    super().__init__(**kwargs)

    # ----------------------------------------------------------------------------------------------
    # Parameters

    self.params = p = utt.containers.ParameterManager()
    p.config('zero', ['*'], 0., False)  # Parameters default to 0, unless initialized otherwise.
    osc_kwargs, basic_kwargs = init(p)

    # ----------------------------------------------------------------------------------------------
    # Circuit units

    # Oscillator units. Left limbs adapt 5% slower to break the initial symmetry.
    XF = -20
    XH = -70
    XFLX = 0
    YFLX = 20
    XEXT = 0
    YEXT = 23
    osc_kwargs_L = {**osc_kwargs, 'adaptation_time': osc_kwargs['adaptation_time'] * 1.05}
    for limb, X, Y, kw in (
      ('FL', XF, +1, osc_kwargs_L),
      ('FR', XF, -1, osc_kwargs),
      ('HL', XH, +1, osc_kwargs_L),
      ('HR', XH, -1, osc_kwargs),
    ):
      setattr(self, f'flx_{limb}', LegacyOscillator(
        bias=p[f'bias_flx_{limb}'], **kw, xy=(X - XFLX, Y * YFLX), anchor='y+'
      ))
      setattr(self, f'ext_{limb}', cc.Basic(
        bias=p[f'bias_ext_{limb}'], **basic_kwargs, xy=(X - XEXT, Y * YEXT)
      ))

    # Interneuron units.
    XC_V0D = XFLX
    XC_V3F = XFLX - 5
    XC_V3E = XEXT - 5
    YC_V0D = YFLX - 13
    YC_V3F = YFLX - 11
    YC_V3E = YEXT + 4
    XD_V0D = 12
    XD_V0V = 12
    XD_IN2 = 6
    YD_V0D = 7
    YD_V0V = 11
    YD_IN2 = 5
    XA_V3A = -12
    YA_V3A = 11
    for limb, X, Y in (('FL', XF, +1), ('FR', XF, -1), ('HL', XH, +1), ('HR', XH, -1)):
      setattr(self, f'cross_v0d_{limb}', cc.Basic(
        bias=p[f'bias_v0d_{limb}'], **basic_kwargs, xy=(X - XC_V0D, Y * YC_V0D), flow='x-',
        anchor='y+'
      ))
      setattr(self, f'cross_v3f_{limb}', cc.Basic(
        bias=p[f'bias_v3f_{limb}'], **basic_kwargs, xy=(X - XC_V3F, Y * YC_V3F), flow='x-',
        anchor='y+'
      ))
      setattr(self, f'cross_v3e_{limb}', cc.Basic(
        bias=p[f'bias_v3e_{limb}'], **basic_kwargs, xy=(X - XC_V3E, Y * YC_V3E), flow='x-'
      ))
      if limb[0] == 'F':
        setattr(self, f'diagd_v0d_{limb}', cc.Basic(
          bias=p[f'bias_v0d_{limb}'], **basic_kwargs, xy=(X - XD_V0D, Y * YD_V0D), anchor='y+'
        ))
        setattr(self, f'diagd_v0v_{limb}', cc.Basic(
          bias=p[f'bias_v0v_{limb}'], **basic_kwargs, xy=(X - XD_V0V, Y * YD_V0V), anchor='y+'
        ))
        setattr(self, f'diagd_in2_{limb}', cc.Basic(
          bias=p[f'bias_in2_{limb}'], **basic_kwargs, xy=(X - XD_IN2, Y * YD_IN2), anchor='y+'
        ))
      else:
        setattr(self, f'diaga_v3a_{limb}', cc.Basic(
          bias=p[f'bias_v3a_{limb}'], **basic_kwargs, xy=(X - XA_V3A, Y * YA_V3A), flow='x-',
          anchor='y+'
        ))

    # Afferent units.
    XAF_FLX = XFLX - 5
    YAF_FLX = YFLX - 5
    YAF_FLXH = YFLX - 7
    XAF_EXT = XEXT - 5
    YAF_EXT = YEXT + 2
    for limb, X, Y in (('FL', XF, +1), ('FR', XF, -1), ('HL', XH, +1), ('HR', XH, -1)):
      yflx = YAF_FLX if limb[0] == 'F' else YAF_FLXH
      setattr(self, f'aff_flx_{limb}', cc.Signal(xy=(X - XAF_FLX, Y * yflx), anchor='y+'))
      setattr(self, f'aff_ext_{limb}', cc.Signal(xy=(X - XAF_EXT, Y * YAF_EXT)))

    # Command units.
    XCMD = -1
    YFREQ = 3
    YSYNC = 1
    self.freq_R = cc.Signal(xy=(XCMD, -YFREQ), anchor='y+')
    self.sync_R = cc.Signal(xy=(XCMD, -YSYNC))
    self.freq_L = cc.Signal(xy=(XCMD, +YFREQ), anchor='y+')
    self.sync_L = cc.Signal(xy=(XCMD, +YSYNC))

    # ----------------------------------------------------------------------------------------------
    # Circuit connections

    # Alias self to ensure that named access of circuit units typechecks correctly.
    u = T.cast(T.Mapping[str, T.Any], self)

    # Oscillator connections (osc).
    for a in LIMBS:
      u[f'flx_{a}'].synapse(u[f'ext_{a}'], weight=p[f'osc_extflx_i_{a}'], sign='-')
      u[f'ext_{a}'].synapse(u[f'flx_{a}'], weight=p[f'osc_flxext_i_{a}'], sign='-')

    # Cross-side connections (cross).
    for a, b in (('FR', 'FL'), ('FL', 'FR'), ('HR', 'HL'), ('HL', 'HR')):
      u[f'cross_v3f_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v3e_{a}'].synapse(u[f'ext_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(u[f'cross_v0d_{a}'], weight=p[f'cross_flxflx_i_{b}'], sign='-')
      u[f'flx_{b}'].synapse(u[f'cross_v3f_{a}'], weight=p[f'cross_flxflx_e_{b}'], sign='+')
      u[f'ext_{b}'].synapse(
        u[f'cross_v3e_{a}'], weight=p[f'cross_extext_e_{b}'], sign='+', waypoints=[(-8, -1, 'y-')]
      )

    # Same-side, descending connections (sided).
    for a, b in (('FL', 'HL'), ('FR', 'HR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sided_extflx_e_{b}'], sign='+', waypoints=[(-10, -2, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'flx_{a}'], weight=p[f'sided_flxflx_i_{b}'], sign='-', waypoints=[(-11, +2, 'y+')]
      )

    # Same-side, ascending connections (sidea).
    for a, b in (('HL', 'FL'), ('HR', 'FR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sidea_extflx_e_{b}'], sign='+', waypoints=[(-3, -7, 'y-')]
      )

    # Diagonal, descending connections (diagd).
    for a, b in (('FL', 'HR'), ('FR', 'HL')):
      u[f'diagd_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'diagd_v0v_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0d_{a}'], weight=p[f'diagd_flxflx_i_{b}'], sign='-', waypoints=[(-35, +12, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0v_{a}'], weight=p[f'diagd_flxflx_e_{b}'], sign='+', waypoints=[(-34, +12, 'y-')]
      )
      u[f'diagd_in2_{a}'].synapse(u[f'cross_v3f_{a}'], weight=1, sign='+')
      u[f'diagd_v0d_{a}'].synapse(u[f'diagd_in2_{a}'], weight=p[f'diagd_in2v0d_i_{a}'], sign='-')

    # Diagonal, ascending connections (diaga).
    for a, b in (('HL', 'FR'), ('HR', 'FL')):
      u[f'diaga_v3a_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diaga_v3a_{a}'], weight=p[f'diaga_flxflx_e_{b}'], sign='+', waypoints=[(32, 7.5, 'x-')]
      )
      u[f'diagd_in2_{b}'].synapse(
        u[f'diaga_v3a_{a}'], weight=p[f'diaga_v3ain2_e_{b}'], sign='+', waypoints=[(26, -7.5, 'x-')]
      )

    # Afferent connections (aff).
    for a in LIMBS:
      u[f'flx_{a}'].synapse(u[f'aff_flx_{a}'], weight=p[f'aff_flx_{a}'])
      u[f'ext_{a}'].synapse(u[f'aff_ext_{a}'], weight=p[f'aff_ext_{a}'])

    # Command connections (freq, sync).
    for a in ('L', 'R'):
      u[f'flx_F{a}'].synapse(
        u[f'freq_{a}'],
        weight=p[f'freq_osc_e_{a}'],
        noise=0.08,
        sign='+',
        waypoints=[(-10, 12, 'y-')],
      )
      u[f'flx_H{a}'].synapse(
        u[f'freq_{a}'],
        weight=p[f'freq_osc_e_{a}'],
        noise=0.08,
        sign='+',
        waypoints=[(-10, 12, 'y-')],
      )
      u[f'ext_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_{a}'], sign='+', waypoints=[(-10, -15, 'y-')]
      )
      u[f'ext_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_{a}'], sign='+', waypoints=[(-9, -15, 'y-')]
      )
      u[f'cross_v3f_F{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'cross_v3f_H{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'diaga_v3a_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_diaga_e_{a}'], sign='+', waypoints=[(-2, 3, 'y-')]
      )
      u[f'cross_v0d_F{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'cross_v0d_H{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'diagd_v0d_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagdi_i_{a}'], sign='-', waypoints=[(-2, 3.5, 'y-')]
      )
      u[f'diagd_v0v_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagde_i_{a}'], sign='-', waypoints=[(-2, 3, 'y-')]
      )

  def get_output_units(self) -> list[str]:
    """Returns the names of the output units, per limb `[flx, ext]`."""
    return [f'{role}_{limb}' for limb in LIMBS for role in ('flx', 'ext')]

  def get_output(self) -> dict[LimbName, torch.Tensor]:
    """Returns each limb's `[flx, ext]` output values, {limb: [2]}."""
    return {
      limb: torch.concat((self[f'flx_{limb}'].value, self[f'ext_{limb}'].value), dim=-1)
      for limb in LIMBS
    }


# ==================================================================================================
# Pattern Formation and Afferent Feedback modules


class _SharedCrossHeads(torch.nn.Module):
  """Linear heads for the front (F) and hind (H) limbs, each shared between the left and right limb
  (the original `share-cross` mode)."""
  def __init__(self, activation: T.Callable[[], torch.nn.Module], bias: bool):
    """Initializes the heads' activation (from its factory) and whether they have biases."""
    super().__init__()
    self.activation = activation()
    self.bias = bias
    self.heads = torch.nn.ModuleDict()

  def initialize(self, input_sizes: T.Mapping[str, int], output_sizes: T.Mapping[str, int]):
    """Creates the zero-initialized heads, with the input and output sizes of each side."""
    for side in ('F', 'H'):
      self.heads[side] = torch.nn.Linear(input_sizes[side], output_sizes[side], bias=self.bias)
      mlp.init_zeros(self.heads[side])

  def forward(
    self,
    limb: LimbName,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Returns `activation(head(x) + bias)` with the head of the limb's side."""
    y = self.heads[limb[0]](x)
    return self.activation(y if bias is None else y + bias)


class QuadrupedPatternFormationNetwork(torch.nn.Module):
  """A Pattern Formation (PF) network generates the muscle activations of each limb from that limb's
  RG [flx, ext] outputs, repeated `repeat` times (the original ignores all observations)."""
  def __init__(
    self,
    flx: T.Mapping[str, T.Mapping[str, T.Sequence[float]]],
    ext: T.Mapping[str, T.Mapping[str, T.Sequence[float]]],
    repeat: int = 8,
    bias: bool = True,
    bias_init: float | T.Mapping[str, T.Sequence[float]] = 0.,
    tie_bias: bool = False,
    activation: T.Callable[[], torch.nn.Module] = torch.nn.Tanh,
    constrain: T.Callable[[torch.nn.Parameter | torch.Tensor], None] | None = ncap.constrain_sign,
    action_prefixes: T.Sequence[str] = ('body/muscle_flx', 'body/muscle_ext'),
  ):
    """Initializes the network.

    Args:
      flx: Initial weights from the RG flexor unit, as {side: {action_prefix: per-joint weights}}
        for side F (front) and H (hind), e.g.
        `{F: {body/muscle_flx: [...], body/muscle_ext: [...]}}`.
      ext: Initial weights from the RG extensor unit, in the same format.
      repeat: Number of redundant copies of each RG input.
      bias: Whether the heads have biases.
      bias_init: Initial value of all biases (the original initializes them to 0), or per-joint
        values as {side: [...]} for side F and H, shared by all action prefixes (e.g. the neutral
        pose of position setpoints).
      tie_bias: Whether each joint shares one bias between its action prefixes (e.g. equal flx and
        ext biases, so no resting differential activation, as the modern `act` biases).
      activation: Output activation.
      constrain: Weight constraint initializer (the original constrains weight signs).
      action_prefixes: Action prefixes, giving action keys f'{prefix}_limb{limb}'.
    """
    super().__init__()
    self.flx = {s: dict(v) for s, v in flx.items()}
    self.ext = {s: dict(v) for s, v in ext.items()}
    self.repeat = repeat
    self.bias_init = bias_init
    self.tie_bias = tie_bias and bias
    self.constrain = constrain
    self.action_prefixes = list(action_prefixes)
    self.net = _SharedCrossHeads(activation, bias and not tie_bias)
    self.biases = torch.nn.ParameterDict()

  def initialize(self, action_space: gymnasium.spaces.Dict):
    """Creates the heads for the action sizes, initializes their weights from `flx` and `ext`
    (split over the `repeat` copies) and their biases, and applies `constrain` to the weights.

    Args:
      action_space: Action space, with every `{prefix}_limb{limb}` key. Contralateral limbs must
        have the same sizes.
    """
    # Action sizes per limb, keyed by limb name (e.g. 'FL').
    self.action_sizes: dict[str, list[int]] = {
      limb: [
        action_space[f'{p}_limb{limb}'].shape[-1]  # type: ignore
        for p in self.action_prefixes
      ]
      for limb in LIMBS
    }
    self.net.initialize(
      {s: 2 * self.repeat for s in ('F', 'H')},
      {s: sum(self.action_sizes[f'{s}L']) for s in ('F', 'H')},
    )
    with torch.no_grad():
      for side in ('F', 'H'):
        head = T.cast(torch.nn.Linear, self.net.heads[side])
        flx = torch.cat([
          torch.as_tensor(self.flx[side][p], dtype=torch.float) for p in self.action_prefixes
        ])
        ext = torch.cat([
          torch.as_tensor(self.ext[side][p], dtype=torch.float) for p in self.action_prefixes
        ])
        assert flx.shape == ext.shape == (head.out_features,), (side, flx.shape, head.out_features)
        head.weight[:, :self.repeat] = flx.reshape(-1, 1) / self.repeat
        head.weight[:, self.repeat:] = ext.reshape(-1, 1) / self.repeat
        if head.bias is not None:
          head.bias[:] = torch.cat([
            self._bias_init(side, n) for n in self.action_sizes[f'{side}L']
          ])
        if self.constrain:
          self.constrain(head.weight)
      if self.tie_bias:
        for side in ('F', 'H'):
          sizes = self.action_sizes[f'{side}L']
          assert len(set(sizes)) == 1, f'Tied biases need equal action sizes, got {sizes}'
          self.biases[side] = torch.nn.Parameter(self._bias_init(side, sizes[0]))

  def _bias_init(self, side: str, size: int) -> torch.Tensor:
    """Initial biases of one action prefix of a side."""
    if isinstance(self.bias_init, (int, float)):
      return torch.full((size,), float(self.bias_init))
    bias = torch.as_tensor(self.bias_init[side], dtype=torch.float)
    assert bias.shape == (size,), (side, bias.shape, size)
    return bias.clone()

  def forward(self, inputs: T.Mapping[LimbName, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Computes the actions of one sample from the RG outputs.

    Args:
      inputs: RG `[flx, ext]` outputs per limb, {limb: [2]}.

    Returns:
      Actions {f'{prefix}_limb{limb}': [size]}.
    """
    actions: dict[str, torch.Tensor] = {}
    for limb in LIMBS:
      # Increase the input size with redundant copies, e.g. [flx, ext] with repeat=3 yields
      # [flx, flx, flx, ext, ext, ext].
      x = inputs[limb].repeat_interleave(self.repeat, dim=-1)
      bias = self.biases[limb[0]].repeat(len(self.action_prefixes)) if self.tie_bias else None
      y = self.net(limb, x, bias)
      y_prefixes = torch.split(y, self.action_sizes[limb], dim=-1)
      for prefix, y_prefix in zip(self.action_prefixes, y_prefixes, strict=True):
        actions[f'{prefix}_limb{limb}'] = y_prefix
    return actions


class QuadrupedAfferentFeedbackNetwork(torch.nn.Module):
  """An Afferent Feedback (AF) network provides [flx, ext] sensory feedback to the RG from each
  limb's rectified observations, through `repeat` redundant outputs that are aggregated."""
  def __init__(
    self,
    observation_prefixes: T.Sequence[str],
    flx: T.Mapping[str, T.Sequence[float]],
    ext: T.Mapping[str, T.Sequence[float]],
    repeat: int = 8,
    repeat_agg: T.Literal['mean', 'sum'] = 'sum',
    scale: float = 1.0,
    constrain: T.Callable[[torch.nn.Parameter | torch.Tensor], None] | None = ncap.constrain_sign,
  ):
    """Initializes the network.

    Args:
      observation_prefixes: Observation prefixes, giving observation keys f'{prefix}_limb{limb}'.
      flx: Initial weights to the RG flexor unit, as {side: weights} for side F and H, over the
        rectified observations [o1-, o2-, ..., o1+, o2+, ...].
      ext: Initial weights to the RG extensor unit, in the same format.
      repeat: Number of redundant outputs per RG unit.
      repeat_agg: Aggregation over redundant outputs.
      scale: Scale of the initial weights.
      constrain: Weight constraint initializer (the original constrains weight signs).
    """
    super().__init__()
    assert repeat_agg in ('mean', 'sum')
    self.observation_prefixes = list(observation_prefixes)
    self.flx = dict(flx)
    self.ext = dict(ext)
    self.repeat = repeat
    self.repeat_agg = repeat_agg
    self.scale = scale
    self.constrain = constrain
    self.net = _SharedCrossHeads(torch.nn.Identity, bias=False)

  def initialize(self, observation_space: gymnasium.spaces.Dict):
    """Creates the heads for the observation sizes, initializes their weights from `flx` and `ext`
    (split over the `repeat` outputs, times `scale`), and applies `constrain` to them.

    Args:
      observation_space: Observation space, with every `{prefix}_limb{limb}` key. Contralateral
        limbs must have the same sizes.
    """
    size = {
      s: 2 * sum(
        observation_space[f'{p}_limb{s}L'].shape[-1]  # type: ignore
        for p in self.observation_prefixes
      )
      for s in ('F', 'H')
    }
    self.net.initialize(size, {s: 2 * self.repeat for s in ('F', 'H')})
    with torch.no_grad():
      for side in ('F', 'H'):
        head = T.cast(torch.nn.Linear, self.net.heads[side])
        flx = torch.as_tensor(self.flx[side], dtype=torch.float)
        ext = torch.as_tensor(self.ext[side], dtype=torch.float)
        assert flx.shape == ext.shape == (head.in_features,), (side, flx.shape, head.in_features)
        head.weight[:self.repeat, :] = flx.reshape(1, -1) / self.repeat * self.scale
        head.weight[self.repeat:, :] = ext.reshape(1, -1) / self.repeat * self.scale
        if self.constrain:
          self.constrain(head.weight)

  def forward(self, observation: T.Mapping[str, torch.Tensor]) -> dict[LimbName, torch.Tensor]:
    """Computes the afferent signals of one sample.

    Args:
      observation: Observations, with every `{prefix}_limb{limb}` key, {key: [size]}.

    Returns:
      `[flx, ext]` afferent signals per limb, {limb: [2]}.
    """
    outputs: dict[LimbName, torch.Tensor] = {}
    for limb in LIMBS:
      x = torch.cat([observation[f'{p}_limb{limb}'] for p in self.observation_prefixes], dim=-1)
      # Split into non-negative negative and positive components, e.g. [a, b] -> [a-, b-, a+, b+].
      x = torch.cat((torch.clamp(-x, min=0), torch.clamp(x, min=0)), dim=-1)
      y = self.net(limb, x)
      # Aggregate over redundant copies, e.g. [flx, flx, flx, ext, ext, ext] -> [flx, ext].
      y = y.reshape(*y.shape[:-1], 2, self.repeat)
      outputs[limb] = y.sum(dim=-1) if self.repeat_agg == 'sum' else y.mean(dim=-1)
    return outputs


class QuadrupedBrainstemCommand(torch.nn.Module):
  """A Brainstem Command (BC) module that writes a fixed affine function of an observation (the
  original writes a constant) into every RG command signal [freq_L, sync_L, freq_R, sync_R]. The
  weight and bias are fixed (not trained)."""
  def __init__(self, observation_key: str = 'target_vx', weight: float = 1.0, bias: float = 0.0):
    """Initializes the command `weight * observation[observation_key] + bias`."""
    super().__init__()
    self.observation_key = observation_key
    self.weight = weight
    self.bias = bias

  def forward(self, observation: T.Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Returns the command of one sample, [1], from the first value of the observation."""
    return observation[self.observation_key][..., :1] * self.weight + self.bias


# ==================================================================================================
# QuadrupedNCAPActor


class QuadrupedNCAPActor(torch.nn.Module):
  """Coordinates the RG, PF, AF, and BC modules as in the original `QuadrupedModularActor`. Body
  actions that the PF does not drive are set to zero.

  The actor is unbatched (map it over a batch with `torch.vmap`). Each step, the AF signals and the
  BC command are written into the RG, the RG steps (signals written in this step take effect from
  the next step), and the PF maps the RG outputs to the actions.
  """
  def __init__(
    self,
    pattern_formation: QuadrupedPatternFormationNetwork,
    rhythm_generation: QuadrupedRhythmGenerationCircuit,
    afferent_feedback: QuadrupedAfferentFeedbackNetwork | None = None,
    brainstem_command: QuadrupedBrainstemCommand | None = None,
    constrainer: 'ncap.Constrainer | None' = None,
    timestep: float = 30.,  # Match to task.control_timestep = 0.03s = 30ms.
  ):
    """Initializes the actor (see `quadruped_es` for the arguments)."""
    super().__init__()
    self.pattern_formation = pattern_formation
    self.sim = cc.Simulator(timestep=timestep)
    self.sim.rhythm_generation = rhythm_generation
    self.afferent_feedback = afferent_feedback
    self.brainstem_command = brainstem_command
    self.constrainer = constrainer

  def initialize(
    self,
    observation_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Dict,
    observation_normalizer: ttm.ObservationNormalizer | None = None,
  ):
    """Initializes the modules for the spaces, and applies the parameter constraints once.

    Args:
      observation_space: Raw observation space of one sample.
      action_space: Action space of one sample. The actions are returned in its key order.
      observation_normalizer: Unused: the actor takes raw observations.
    """
    # The actor takes raw observations: an actor-critic model's normalizer is for its critic only.
    self.action_space = action_space
    self.sim.init()
    self.pattern_formation.initialize(action_space)
    if self.afferent_feedback is not None:
      self.afferent_feedback.initialize(observation_space)
    if self.constrainer:
      self.constrainer.initialize(self.named_parameters())
      self.constrainer.constrain()

  def reset(self, reset: torch.Tensor | None = None):
    """Resets the RG (where `reset` is true, if given)."""
    self.sim.reset(reset)

  def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Returns the actions of one sample, in the action space's key order (see the class)."""
    rg = T.cast(QuadrupedRhythmGenerationCircuit, self.sim.rhythm_generation)
    if self.afferent_feedback is not None:
      # Write afferent signals into sim.
      afferents = self.afferent_feedback(observation)
      for limb in LIMBS:
        rg[f'aff_flx_{limb}'].value[:] = afferents[limb][..., 0]
        rg[f'aff_ext_{limb}'].value[:] = afferents[limb][..., 1]
    if self.brainstem_command is not None:
      # Write the command into every RG command signal.
      command = self.brainstem_command(observation)
      for signal in (rg.freq_L, rg.sync_L, rg.freq_R, rg.sync_R):
        signal.value[:] = command

    # Any signals written on this timestep will be incorporated at the next timestep.
    self.sim.step()

    actions = self.pattern_formation(rg.get_output())
    return collections.OrderedDict(
      (k, actions[k] if k in actions else torch.zeros(self.action_space[k].shape))  # type: ignore
      for k in self.action_space.keys()
    )
