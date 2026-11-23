import functools
import typing as T

import gymnasium
import numpy as np
import pytest
import torch

import circuits as cc
import composer.utils.gym as cxg
import composer.utils.torch as cxt
from models import ncap

LIMBS = ('limbFL', 'limbFR', 'limbHL', 'limbHR')
LATERAL = {'F': ('limbFL', 'limbFR'), 'H': ('limbHL', 'limbHR')}
FLX_DELAYS = [0.25, 0.0, 0.2, 0.0]  # As `model/rat/rhythm_generation/rg.yaml`.


def init_rg_rat_noiseless(p, grad: bool = False):
  """`init_rg_rat` (scale 0.8) without the flexors' noise."""
  flx, ext, basic = ncap.init_rg_rat(p, grad, scale=0.8)
  return {**flx, 'noise': 0.}, ext, basic


def make_rhythm_generation(noise: bool = True) -> ncap.RhythmGenerationCircuit:
  """The provided RG config: `init_rg_rat` (scale 0.8), Basic extensors, its flexor delays."""
  return ncap.RhythmGenerationCircuit(
    init=functools.partial(ncap.init_rg_rat, scale=0.8) if noise else init_rg_rat_noiseless,
    ext_oscillator=False,
    flx_delays=FLX_DELAYS,
  )


def make_spaces() -> tuple[gymnasium.spaces.Dict, gymnasium.spaces.Dict]:
  """Small observation and action spaces: a speed command and 3 joint positions per limb; 2 flexor
  and 2 extensor muscles per limb."""
  observation_space = gymnasium.spaces.Dict({
    'target_vx': cxg.Box(-1., 1., shape=(1,)),
    **{f'body/proprio_pos_{limb}': cxg.Box(-1., 1., shape=(3,)) for limb in LIMBS},
  })
  action_space = gymnasium.spaces.Dict({
    f'body/muscle_{prefix}_{limb}': cxg.Box(0., 1., shape=(2,))
    for prefix in ('flx', 'ext') for limb in LIMBS
  })
  return observation_space, action_space


def make_pattern_formation() -> ncap.SynergyPatternFormationNetwork:
  """A PF of 2 synergies per limb from the RG outputs, joint positions and speed command."""
  synergies = {
    f'{group}_{i}': {
      'body/muscle_flx': f'{group}_{i}_flx', 'body/muscle_ext': f'{group}_{i}_ext',
    }
    for group in ('F', 'H') for i in (1, 2)
  }
  return ncap.SynergyPatternFormationNetwork(
    observation_prefixes={
      **{limb: ['rg', 'body/proprio_pos'] for limb in LIMBS}, 'bc': ['speed'],
    },
    action_segments=list(LIMBS),
    action_prefixes=['body/muscle_flx', 'body/muscle_ext'],
    synergies=synergies,
    segment_synergies={
      'limbFL': ['F_1', 'F_2'], 'limbFR': ['F_1', 'F_2'],
      'limbHL': ['H_1', 'H_2'], 'limbHR': ['H_1', 'H_2'],
    },
    action_synergies={limb: [limb] for limb in LIMBS},
    lateral_segments=LATERAL,
    sagittal_segments={'BCs': 'bc'},
    observation_prefix_shorthand={'rg': 'rg', 'body/proprio_pos': 'pos', 'speed': 'speed'},
    action_prefix_shorthand={'body/muscle_flx': 'flx', 'body/muscle_ext': 'ext'},
    select_activation=functools.partial(torch.nn.Hardtanh, 0., 1.),
    synergy_activation=functools.partial(torch.nn.Hardtanh, 0., 1.),
  )


def make_afferent_feedback() -> ncap.AfferentFeedbackNetwork:
  """An AF from each limb's joint positions to the RG's afferent signals of the limbs."""
  return ncap.AfferentFeedbackNetwork(
    observation_prefixes={limb: ['body/proprio_pos'] for limb in LIMBS},
    action_segments=list(LIMBS),
    action_prefixes=['flx', 'ext'],
    lateral_segments=LATERAL,
    observation_prefix_shorthand={'body/proprio_pos': 'pos'},
    reset=True,
  )


def fill_parameters(module: torch.nn.Module, seed: int = 0) -> None:
  """Sets every trainable parameter of `module` to small random values."""
  generator = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for param in module.parameters():
      if param.requires_grad:
        param[:] = 0.5 * torch.randn(param.shape, generator=generator)


def make_actor(noise: bool = True) -> ncap.RatNcapActor:
  """A small rat NCAP actor (see the `make_*` helpers), initialized with random PF/AF weights."""
  actor = ncap.RatNcapActor(
    pattern_formation=make_pattern_formation(),
    afferent_feedback=make_afferent_feedback(),
    rhythm_generation=make_rhythm_generation(noise),
    brainstem_command=ncap.BrainstemCommandNetwork(
      init=functools.partial(ncap.init_bc_fixed, weight=1., bias=0.),
    ),
    observation_rectifier=ncap.ObservationRectifier({'body/proprio_pos*': 'flip'}, weight=False,
                                                    bias=False),
  )
  actor.initialize(*make_spaces())
  fill_parameters(actor)
  return actor


def observations(steps: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
  """Random observations of the small spaces, at speed 0.4."""
  generator = torch.Generator().manual_seed(seed)
  observation_space, _ = make_spaces()
  return [
    {
      key: (torch.full((1,), 0.4) if key == 'target_vx' else
            2 * torch.rand(T.cast(tuple[int, ...], space.shape), generator=generator) - 1)
      for key, space in observation_space.items()
    }
    for _ in range(steps)
  ]


# ==================================================================================================
# RatNcapActor


def test_actor_requires_afferent_feedback():
  with pytest.raises(AssertionError, match='afferent feedback'):
    ncap.RatNcapActor(
      pattern_formation=make_pattern_formation(),
      afferent_feedback=None,  # type: ignore
      rhythm_generation=make_rhythm_generation(),
    )


def test_actor_returns_every_action_in_the_action_space_order():
  actor = make_actor()
  actor.reset()

  action = actor(observations(1)[0])

  _, action_space = make_spaces()
  assert list(action) == list(action_space)
  assert all(action[key].shape == space.shape for key, space in action_space.items())


def test_actor_vmapped_over_samples_matches_each_sample_alone():
  actor = make_actor(noise=False)
  samples = [observations(20, seed=seed) for seed in (1, 2)]
  params: dict[str, torch.Tensor] = dict(actor.named_parameters())

  def run_alone(steps: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Runs the actor on one sample's steps from a reset."""
    actor.reset()
    return [actor(observation) for observation in steps]

  def step(buffers, observation):
    """One functional step of the actor."""
    return torch.func.functional_call(actor, (params, buffers), args=(observation,))

  expected = [run_alone(steps) for steps in samples]
  actor.reset()
  buffers = {k: v.clone().repeat(2, *[1] * v.ndim) for k, v in actor.named_buffers()}
  for t in range(20):
    batch = {k: torch.stack([samples[0][t][k], samples[1][t][k]]) for k in samples[0][t]}
    # The noiseless oscillators still draw (zero-scaled) noise, so vmap needs a randomness mode.
    actions = torch.func.vmap(step, randomness='same')(buffers, batch)
    for i in range(2):
      for key, value in actions.items():
        torch.testing.assert_close(value[i], expected[i][t][key])


def test_reset_restores_the_rhythm_generation_only_where_reset_is_true():
  actor = make_actor()
  actor.reset()
  initial = {k: v.clone() for k, v in actor.named_buffers()}
  for observation in observations(10):
    actor(observation)
  stepped = {k: v.clone() for k, v in actor.named_buffers()}

  actor.reset(torch.tensor([False]))
  kept = {k: v.clone() for k, v in actor.named_buffers()}
  actor.reset(torch.tensor([True]))

  for name, value in actor.named_buffers():
    torch.testing.assert_close(kept[name], stepped[name], rtol=0, atol=0)
    if 'aff_' not in name and 'freq_' not in name and 'sync_' not in name:  # Signals: written.
      torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_pack_and_unpack_round_trip_the_trainable_parameters():
  actor = make_actor()
  params: dict[str, torch.Tensor] = dict(actor.named_parameters())
  trainable = {k for k, v in params.items() if v.requires_grad}
  original = {k: v.detach().clone() for k, v in params.items()}

  vector = cxt.pack_tensors(params, grad_only=True)
  with torch.no_grad():
    for k in trainable:
      params[k].zero_()
  cxt.unpack_tensors(vector, params, grad_only=True)

  assert vector.shape == (sum(original[k].numel() for k in trainable),)
  for name, value in params.items():
    torch.testing.assert_close(value.detach(), original[name], rtol=0, atol=0)
  assert not any(k.startswith('sim.') for k in trainable)  # The RG is frozen.


# ==================================================================================================
# SynergyPatternFormationNetwork


def initialized_pattern_formation() -> ncap.SynergyPatternFormationNetwork:
  """The small PF, initialized for its spaces, with random parameters."""
  pattern_formation = make_pattern_formation()
  observation_space = gymnasium.spaces.Dict({
    **{f'rg_{limb}': cxg.Box(0., 1., shape=(4,)) for limb in LIMBS},
    **{f'body/proprio_pos_{limb}': cxg.Box(-1., 1., shape=(3,)) for limb in LIMBS},
    'speed_bc': cxg.Box(-1., 1., shape=(1,)),
  })
  pattern_formation.initialize(observation_space, make_spaces()[1])
  fill_parameters(pattern_formation)
  return pattern_formation


def test_pattern_formation_mirrors_left_and_right_limbs():
  pattern_formation = initialized_pattern_formation()
  generator = torch.Generator().manual_seed(1)
  observation = {key: torch.rand(size, generator=generator) for key, size in
                 [*((f'rg_{limb}', 4) for limb in LIMBS),
                  *((f'body/proprio_pos_{limb}', 3) for limb in LIMBS), ('speed_bc', 1)]}
  swap = {'limbFL': 'limbFR', 'limbFR': 'limbFL', 'limbHL': 'limbHR', 'limbHR': 'limbHL'}
  mirrored = {
    key: observation[key.replace(key.split('_')[-1], swap[key.split('_')[-1]])]
    if key.split('_')[-1] in swap else value
    for key, value in observation.items()
  }

  action = pattern_formation(observation)
  mirrored_action = pattern_formation(mirrored)

  for key, value in action.items():
    limb = key.split('_')[-1]
    torch.testing.assert_close(mirrored_action[key.replace(limb, swap[limb])], value)


def test_pattern_formation_shares_synergy_weights_between_lateral_limbs():
  pattern_formation = initialized_pattern_formation()

  names = [name for name, _ in pattern_formation.named_parameters()]

  assert any('F_1_flx' in name for name in names)
  assert not any('limbFL' in name or 'limbFR' in name for name in names)  # Shared by group.


def test_pattern_formation_rejects_undefined_synergies():
  with pytest.raises(AssertionError, match='Undefined synergies'):
    ncap.SynergyPatternFormationNetwork(
      observation_prefixes={'limbFL': ['rg']},
      action_segments=['limbFL'],
      action_prefixes=['body/muscle_flx'],
      synergies={'F_1': {'body/muscle_flx': 'F_1_flx'}},
      segment_synergies={'limbFL': ['F_1', 'F_2']},
      action_synergies={'limbFL': ['limbFL']},
      lateral_segments={'F': ('limbFL', 'limbFR')},
    )


def test_pattern_formation_ignores_synergies_mapped_to_none():
  pattern_formation = ncap.SynergyPatternFormationNetwork(
    observation_prefixes={'limbFL': ['rg']},
    action_segments=['limbFL'],
    action_prefixes=['body/muscle_flx'],
    synergies={'F_1': {'body/muscle_flx': 'F_1_flx'}, 'F_2': None},
    segment_synergies={'limbFL': ['F_1']},
    action_synergies={'limbFL': ['limbFL']},
    sagittal_segments={'F': 'limbFL'},
  )

  assert pattern_formation.synergy_labels == {'limbFL': ['F_1']}


# ==================================================================================================
# AfferentFeedbackNetwork


def test_afferent_feedback_outputs_sensors_and_limb_resets():
  afferent_feedback = make_afferent_feedback()
  observation_space = gymnasium.spaces.Dict({
    f'body/proprio_pos_{limb}': cxg.Box(-1., 1., shape=(3,)) for limb in LIMBS
  })
  reset_space = gymnasium.spaces.Dict({
    f'{prefix}_{limb}': cxg.Box(0., 1., shape=(1,)) for prefix in ('flx', 'ext') for limb in LIMBS
  })
  afferent_feedback.initialize(observation_space, reset_space)

  sensor, reset = afferent_feedback({key: torch.ones(3) for key in observation_space})

  assert set(sensor) == set(observation_space)
  assert reset is not None and set(reset) == set(reset_space)


# ==================================================================================================
# BodyPlan


@pytest.mark.parametrize(
  'source, target, expected',
  [
    ('limbFL', 'limbFL', 'F>F'),
    ('limbFL', 'limbFR', 'F>Fc'),
    ('limbFR', 'limbHL', 'F>Hc'),
    ('limbFL', 'lumbarE', 'F>Le'),
    ('lumbarBT', 'limbFR', 'Lbt>Fc'),
    ('lumbarBT', 'limbFL', 'Lbt>F'),
  ],
)
def test_intersegment_weight_ids_mark_contralateral_connections(source, target, expected):
  bodyplan = ncap.BodyPlan(LATERAL, {'Le': 'lumbarE'}, {'Lbt': 'lumbarBT'})

  assert bodyplan.intersegment_weight_id(source, target) == expected


def test_classify_segment_rejects_unknown_segments():
  with pytest.raises(ValueError):
    ncap.BodyPlan(LATERAL, {}, {}).classify_segment('tail')


# ==================================================================================================
# Brainstem command


def test_fixed_brainstem_command_passes_the_speed_through():
  brainstem_command = ncap.BrainstemCommandNetwork(
    init=functools.partial(ncap.init_bc_fixed, weight=1., bias=0.),
  )
  brainstem_command.initialize(gymnasium.spaces.Dict({'target_vx': cxg.Box(-1., 1., shape=(1,))}))

  commands = brainstem_command({'target_vx': torch.tensor([0.6])})

  torch.testing.assert_close(commands['speed'], torch.tensor([0.6]))
  assert not any(param.requires_grad for param in brainstem_command.parameters())


# ==================================================================================================
# ObservationRectifier


def rectify(rectification: ncap.RectifyMode, split: bool, **kwargs) -> dict[str, torch.Tensor]:
  """Rectifies `x = [-0.5, 0.25]` of box [-1, 1] with `rectification`."""
  rectifier = ncap.ObservationRectifier({'x': rectification}, split_key={'x': split}, **kwargs)
  rectifier.initialize(gymnasium.spaces.Dict({'x': cxg.Box(-1., 1., shape=(2,))}))
  return rectifier({'x': torch.tensor([-0.5, 0.25])})


def test_flip_mirrors_the_value_within_its_box():
  rectified = rectify('flip', split=True, weight=False, bias=False)

  torch.testing.assert_close(rectified['x[<-]'], torch.tensor([0.5, -0.25]))
  torch.testing.assert_close(rectified['x[->]'], torch.tensor([-0.5, 0.25]))


def test_sign_splits_into_nonnegative_channels():
  rectified = rectify('sign', split=True, weight=False, bias=False)

  torch.testing.assert_close(rectified['x[-]'], torch.tensor([0.5, 0.]))
  torch.testing.assert_close(rectified['x[+]'], torch.tensor([0., 0.25]))


def test_unsplit_channels_are_concatenated_under_the_key():
  rectified = rectify('sign', split=False, weight=False, bias=False)

  torch.testing.assert_close(rectified['x'], torch.tensor([0.5, 0., 0., 0.25]))


def test_learnable_affine_starts_as_the_identity():
  rectified = rectify('sign', split=False, weight=True, bias=True)

  torch.testing.assert_close(rectified['x'], torch.tensor([0.5, 0., 0., 0.25]))


def test_unmatched_activation_patterns_are_rejected():
  rectifier = ncap.ObservationRectifier({'x': 'flip'}, activation={'y*': torch.nn.ReLU})

  with pytest.raises(ValueError, match='match no observation keys'):
    rectifier.initialize(gymnasium.spaces.Dict({'x': cxg.Box(-1., 1., shape=(2,))}))


# ==================================================================================================
# RhythmGenerationCircuit


def simulate_rhythm_generation(speed: float, steps: int = 400) -> np.ndarray:
  """Simulates the provided RG at a constant BC speed (no afferents), 30 ms steps, and returns its
  outputs, [steps, limb (FL, FR, HL, HR), output (flx, flx2, ext, ext2)]."""
  torch.manual_seed(0)
  sim = cc.Simulator(timestep=30)
  sim.rhythm_generation = rhythm_generation = make_rhythm_generation()
  sim.init()
  sim.reset()
  rhythm_generation.reset()
  outputs = []
  for _ in range(steps):
    for signal in ('freq_L', 'sync_L', 'freq_R', 'sync_R'):
      getattr(rhythm_generation, signal).value[:] = speed
    sim.step()
    output = rhythm_generation.get_output()
    outputs.append(np.stack([output[limb].numpy() for limb in ('FL', 'FR', 'HL', 'HR')]))
  return np.array(outputs)


def burst_onsets(flexor: np.ndarray) -> np.ndarray:
  """Steps at which a flexor output turns active (above 0.05)."""
  active = flexor > 0.05
  return np.where(active[1:] & ~active[:-1])[0] + 1


def phase_offsets(outputs: np.ndarray) -> list[float]:
  """Median burst-onset phase of FR, HL and HR relative to FL, in [0, 1), after a transient."""
  onsets = [burst_onsets(outputs[100:, limb, 0]) for limb in range(4)]
  period = np.mean(np.diff(onsets[0]))
  offsets = []
  for limb in (1, 2, 3):
    delays = [onsets[limb][onsets[limb] >= start][0] - start for start in onsets[0][:-1]]
    offsets.append(float(np.median(np.mod(np.array(delays) / period, 1))))
  return offsets


def test_rhythm_generation_is_quiet_at_zero_speed():
  outputs = simulate_rhythm_generation(0.)

  assert outputs[100:, :, 0].max() < 0.05


def test_rhythm_generation_alternates_flexors_and_extensors_at_positive_speed():
  outputs = simulate_rhythm_generation(0.4)

  flexor_active = outputs[100:, :, 0] > 0.05
  extensor_active = outputs[100:, :, 2] > 0.05
  assert all(len(burst_onsets(outputs[100:, limb, 0])) >= 10 for limb in range(4))
  assert (flexor_active & extensor_active).mean() < 0.15


def test_rhythm_generation_steps_faster_at_higher_speed():
  periods = [np.mean(np.diff(burst_onsets(simulate_rhythm_generation(speed)[100:, 0, 0])))
             for speed in (0.2, 0.4, 1.0)]

  assert periods[0] > periods[1] > periods[2]


@pytest.mark.parametrize(
  'speed, expected',  # Phases of FR, HL and HR relative to FL.
  [(0.2, [0.5, 0.75, 0.25]), (0.4, [0.5, 0.5, 0.]), (1.0, [0., 0.7, 0.7])],
  ids=['walk', 'trot', 'bound'],
)
def test_rhythm_generation_interlimb_phases_follow_the_gait_of_the_speed(speed, expected):
  offsets = phase_offsets(simulate_rhythm_generation(speed))

  distances = [min(abs(o - e), 1 - abs(o - e)) for o, e in zip(offsets, expected)]
  assert max(distances) < 0.1, offsets


def test_flx2_follows_flx_after_two_steps_and_ext2_after_one():
  sim = cc.Simulator(timestep=30)
  sim.rhythm_generation = rhythm_generation = make_rhythm_generation()
  sim.init()

  units = {name: T.cast(cc.Basic, rhythm_generation[f'{name}_FL']) for name in ('flx2', 'ext2')}

  delays = {
    name: [connection.delay.delay for connection in unit.synapses.connections.values()]
    for name, unit in units.items()
  }

  assert delays == {'flx2': [2], 'ext2': [1, 1]}


def test_output_units_are_flx_flx2_ext_ext2_per_limb():
  rhythm_generation = make_rhythm_generation()

  assert rhythm_generation.get_output_units() == [
    f'{unit}_{limb}' for limb in ('FL', 'FR', 'HL', 'HR') for unit in ('flx', 'flx2', 'ext', 'ext2')
  ]


@pytest.mark.parametrize('unit', [
  'cross_v0d_FL', 'cross_v3f_HR', 'cross_v3e_FR', 'diagd_v0d_FL', 'diagd_v0v_FR', 'diagd_in2_FL',
  'diaga_v3a_HL', 'aff_flx_FL', 'aff_ext_HR', 'freq_L', 'sync_R',
])
def test_interneurons_and_signals_keep_their_names(unit):
  assert isinstance(getattr(make_rhythm_generation(), unit), cc.Signal)


def test_init_rg_rat_sets_the_rat_weights_and_freezes_them_by_default():
  rhythm_generation = make_rhythm_generation()
  params = rhythm_generation.params

  assert params['cross_flxflx_i_FR'].item() == pytest.approx(-1.7)
  assert params['cross_flxflx_i_FL'].item() == pytest.approx(-2.5)
  assert params['aff_flx_FL'].item() == 1.
  assert params['unconfigured_name'].item() == 0.  # The frozen `zero` parameter.
  assert not any(param.requires_grad for param in rhythm_generation.parameters())


def test_init_rg_rat_trains_only_the_requested_parameters():
  rhythm_generation = ncap.RhythmGenerationCircuit(
    init=functools.partial(ncap.init_rg_rat, grad_flx=True),
    ext_oscillator=False,
    flx_delays=FLX_DELAYS,
  )
  params = rhythm_generation.params

  assert params['bias_flx_FL'].requires_grad
  assert not params['bias_ext_FL'].requires_grad
  assert not params['cross_flxflx_i_FR'].requires_grad


# ==================================================================================================
# Constrainer


def constrained(values: list[float], **kwargs) -> torch.nn.Parameter:
  """A parameter of `values` with `constrain(**kwargs)` applied."""
  param = torch.nn.Parameter(torch.tensor(values))
  ncap.constrain(param, **kwargs)
  return param


def perturb_and_constrain(params: dict[str, torch.nn.Parameter]) -> None:
  """Sets the parameters to large random values, then constrains them."""
  constrainer = ncap.Constrainer()
  constrainer.initialize(params.items())
  generator = torch.Generator().manual_seed(0)
  with torch.no_grad():
    for param in params.values():
      param[:] = 10 * torch.randn(param.shape, generator=generator)
  constrainer.constrain()


def test_sign_constraints_keep_signs_and_zero_constraints_keep_zeros():
  param = constrained([1., -2., 0., 3., -4., 0.], sign=True, zero=True)

  perturb_and_constrain({'w': param})

  assert (param[[0, 3]] >= 0).all() and (param[[1, 4]] <= 0).all()
  assert (param[[2, 5]] == 0).all()


def test_constraints_on_views_apply_to_the_base_parameter():
  param = torch.nn.Parameter(torch.tensor([1., -1., 1., -1.]))
  ncap.constrain(param, sign=True)
  ncap.constrain(param[:2], freeze=True)

  perturb_and_constrain({'w': param})

  torch.testing.assert_close(param[:2].detach(), torch.tensor([1., -1.]))
  assert param[2] >= 0 and param[3] <= 0


@pytest.mark.parametrize('bound', [{'min': 0.5}, {'max': -0.5}], ids=['min', 'max'])
def test_one_sided_bounds_are_enforced(bound):
  param = torch.nn.Parameter(torch.zeros(4))
  ncap.constrain_bound(param, **bound)

  perturb_and_constrain({'w': param})

  if 'min' in bound:
    assert (param >= 0.5).all()
  else:
    assert (param <= -0.5).all()


def test_unconstrained_parameters_are_not_collected():
  constrainer = ncap.Constrainer()

  constrainer.initialize([('free', torch.nn.Parameter(torch.zeros(2))),
                          ('bounded', constrained([1.], sign=True))])

  assert constrainer.names == ['bounded']
