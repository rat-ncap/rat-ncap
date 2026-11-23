"""
This file defines the rat NCAP architecture, based on neural circuits in the limbs and spinal cord
of quadruped mammals. The actor (`RatNcapActor`) is composed of:

- **Observation Rectifier**: splits signed observations into nonnegative channels.
- **Afferent Feedback** (AF): sensory layer over the (rectified) observations of every limb and body
  segment, and a reset layer that drives the RG's afferent signals of each limb.
- **Brainstem Command** (BC): commands (e.g. speed) from the task's command observations.
- **Rhythm Generation** (RG): a circuit of oscillators and interneurons that coordinates the
  rhythm of the four limbs, driven by the BC speed and the AF reset signals.
- **Pattern Formation** (PF): maps the AF sensor outputs, the RG outputs and the BC commands to
  the actions of every limb and body segment, through shared synergies.
- **Constrainer**: keeps parameters within their sign and magnitude constraints.

Data flow per step: observations -> rectifier -> AF -> {AF sensor outputs -> PF, AF reset -> RG};
observations -> BC -> {RG, PF}; RG outputs -> PF -> actions.
"""
import collections
import functools
import typing as T

import gymnasium
import numpy as np
import torch

import tonic.torch.models as ttm

import circuits as cc
import composer.utils.gym as cxg
import utils.collections as utc
import utils.string as uts
import utils.torch as utt
import utils.torch.block as utb

__all__ = [
  'ncap_es', 'RatNcapActor', 'LinearPatternFormationNetwork', 'SynergyPatternFormationNetwork',
  'AfferentFeedbackNetwork', 'SegmentType', 'BodyPlan', 'init_bc_fixed', 'BrainstemCommandSignal',
  'BrainstemCommandNetwork', 'ObsKey', 'ActKey', 'ConstrainFn', 'RectifyMode', 'RECTIFY_SUFFIX',
  'ObservationRectifier', 'LimbName', 'RhythmGenerationCircuitObserver',
  'RhythmGenerationOscKwargs', 'RhythmGenerationBasicKwargs', 'RhythmGenerationCircuitInitializer',
  'logarithmic_mod_amplitude', 'activation_locomotion', 'activation_locomotion4_flx',
  'activation_locomotion4_ext', 'init_rg_default', 'init_rg_full', 'init_rg_mouse', 'init_rg_rat',
  'RhythmGenerationCircuit', 'Constrainer', 'DictModule', 'constrain_sign',
  'constrain_bound', 'constrain',
]


def ncap_es(
  pattern_formation: 'LinearPatternFormationNetwork | SynergyPatternFormationNetwork',
  afferent_feedback: 'AfferentFeedbackNetwork',
  rhythm_generation: 'RhythmGenerationCircuit | None' = None,
  brainstem_command: 'BrainstemCommandSignal | BrainstemCommandNetwork | None' = None,
  observation_rectifier: 'ObservationRectifier | None' = None,
  constrainer: 'Constrainer | None' = None,
  on_initialize: T.Callable[[ttm.ActorOnly], None] | None = None,
  timestep: float = 30.,
) -> ttm.ActorOnly:
  """Builds the rat NCAP model for evolution strategies: an actor without a critic.

  Args:
    pattern_formation: PF network.
    afferent_feedback: AF network.
    rhythm_generation: RG circuit.
    brainstem_command: BC network, or `None` for no commands.
    observation_rectifier: Observation rectifier, or `None` to use the observations as they are.
    constrainer: Constrainer of the parameters, or `None` for no constraints.
    on_initialize: Function called with the model after it is initialized.
    timestep: Duration of one step of the RG simulation (ms). Must equal the task's control
      timestep (e.g. 30 ms for 0.03 s).

  Returns:
    The model.
  """
  actor: ttm.ActorLike = RatNcapActor(  # type: ignore
    pattern_formation=pattern_formation,
    afferent_feedback=afferent_feedback,
    rhythm_generation=rhythm_generation,
    brainstem_command=brainstem_command,
    observation_rectifier=observation_rectifier,
    constrainer=constrainer,
    timestep=timestep
  )
  return ttm.ActorOnly(actor=actor, on_initialize=on_initialize)


# ==================================================================================================
# RatNcapActor

class RatNcapActor(torch.nn.Module):
  """This top-level module coordinates the interactions between the RG, PF, AF, and BC modules.
  Due to the highly modular structure, different combinations of modules can be included to test
  architectural variants.

  The actor is unbatched: it takes one sample's observations and keeps one sample's RG state. To
  run a batch, map it over the batch with `torch.vmap` (with batched parameters and buffers).

  Each step (`forward`):
  1. The rectifier transforms the raw observations.
  2. The AF computes its sensor outputs and its reset outputs, which are written into the RG's
     afferent signals (`aff_{flx,ext}_<limb>`).
  3. The BC computes its commands; the `speed` command is written into all of the RG's command
     signals (`freq_{L,R}`, `sync_{L,R}`). Other commands go to the PF only.
  4. The RG steps. Signals written in this step take effect from the next step.
  5. The PF maps the AF sensor outputs, the RG outputs (`rg_<limb>`) and the BC commands
     (`<command>_bc`) to the actions.
  """
  def __init__(
    self,
    pattern_formation: 'LinearPatternFormationNetwork | SynergyPatternFormationNetwork',
    afferent_feedback: 'AfferentFeedbackNetwork',
    rhythm_generation: 'RhythmGenerationCircuit | None' = None,
    brainstem_command: 'BrainstemCommandSignal | BrainstemCommandNetwork | None' = None,
    observation_rectifier: 'ObservationRectifier | None' = None,
    constrainer: 'Constrainer | None' = None,
    timestep: float = 30.,  # Match to task.control_timestep = 0.03s = 30ms.
  ):
    """Initializes the actor (see `ncap_es` for the arguments). The AF and the RG are required,
    and the BC requires the RG."""
    super().__init__()

    self.pattern_formation = pattern_formation

    assert afferent_feedback is not None and rhythm_generation is not None, (
      'The actor requires afferent feedback, which requires rhythm generation.'
    )
    self.afferent_feedback = afferent_feedback

    if rhythm_generation is not None:
      self.sim = cc.Simulator(timestep=timestep)
      self.sim.rhythm_generation = rhythm_generation
    else:
      self.sim = None

    assert brainstem_command is None or rhythm_generation is not None
    self.brainstem_command = brainstem_command

    self.observation_rectifier = observation_rectifier
    self.constrainer = constrainer

  def initialize(
    self,
    observation_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Dict,
    observation_normalizer: ttm.ObservationNormalizer | None = None,
  ):
    """Initializes the modules for the spaces, and applies the parameter constraints once.

    The AF's action space is fixed: one `flx_<limb>` and one `ext_<limb>` value per limb, which
    drive the RG. The PF's observation space is the AF's sensor outputs, the RG outputs and the BC
    commands.

    Args:
      observation_space: Raw observation space of one sample.
      action_space: Action space of one sample. The actions are returned in its key order.
      observation_normalizer: Unused: the actor takes raw observations.
    """
    # The actor takes raw observations: an actor-critic model's normalizer is for its critic only.
    self.action_space = action_space

    if self.observation_rectifier is not None:
      observation_space = self.observation_rectifier.initialize(observation_space)

    if self.sim is not None:
      self.sim.init()
      self.rg_observer = RhythmGenerationCircuitObserver()
      observation_space = self.rg_observer.initialize(
        T.cast(RhythmGenerationCircuit, self.sim.rhythm_generation),
        observation_space
      )

      if self.afferent_feedback is not None:
        afferent_feedback_action_space = gymnasium.spaces.Dict({
          f'{prefix}_{segment}': cxg.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
          ) for prefix in ['flx', 'ext']
          for segment in ['limbFL', 'limbFR', 'limbHL', 'limbHR']
        })
        self.afferent_feedback.initialize(observation_space, afferent_feedback_action_space)
      if self.brainstem_command is not None:
        # Brainstem commands are observed by the PF as f'{command}_bc', e.g. `speed_bc`.
        command_space = self.brainstem_command.initialize(observation_space)
        observation_space = gymnasium.spaces.Dict(list(observation_space.items()))
        for command, space in command_space.items():
          observation_space[f'{command}_bc'] = space
    self.pattern_formation.initialize(observation_space, action_space)

    if self.constrainer:
      self.constrainer.initialize(self.named_parameters())
      self.constrainer.constrain()

  def reset(self, reset: torch.Tensor | None = None):
    """Resets the RG state, including its staggered initial limb phases (`sim.init` does not
    apply them, so call this before the first step).

    Args:
      reset: Whether to reset, bool `[1]` (per sample when vmapped), or `None` to reset.
    """
    if self.sim is not None:
      self.sim.reset(reset)
      rg = T.cast(RhythmGenerationCircuit, self.sim.rhythm_generation)
      rg.reset(reset)

  def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Computes the actions of one step (see the class).

    Args:
      observation: Raw observations of one sample, {key: [size]}.

    Returns:
      Actions of one sample, {key: [size]}, in the action space's key order.
    """
    if self.observation_rectifier is not None:
      observation = self.observation_rectifier(observation)

    nonreset_af: dict[str, torch.Tensor]
    reset_af: dict[str, torch.Tensor] | None
    nonreset_af, reset_af = self.afferent_feedback(observation)

    if self.sim is not None:
      rg = T.cast(RhythmGenerationCircuit, self.sim.rhythm_generation)
      if reset_af is not None:
        # Write afferent signals into sim.
        for limb in ('FL', 'FR', 'HL', 'HR'):
          flx = reset_af[f'flx_limb{limb}']
          ext = reset_af[f'ext_limb{limb}']
          T.cast(cc.Signal, rg[f'aff_flx_{limb}']).value[:] = flx
          T.cast(cc.Signal, rg[f'aff_ext_{limb}']).value[:] = ext
      if self.brainstem_command is not None:
        # Write the speed command into every RG command signal.
        commands = self.brainstem_command(observation)
        for signal in (rg.freq_L, rg.sync_L, rg.freq_R, rg.sync_R):
          signal.value[:] = commands['speed'][0]

      # Any signals written on this timestep will be incorporated at the next timestep.
      self.sim.step()
      nonreset_af = self.rg_observer.insert(rg.get_output(), nonreset_af)  # type: ignore
      if self.brainstem_command is not None:
        nonreset_af.update({f'{command}_bc': value for command, value in commands.items()})

    outputs: dict[str, torch.Tensor] = self.pattern_formation(nonreset_af)
    outputs = collections.OrderedDict((k, outputs[k]) for k in self.action_space.keys())
    return outputs


# ==================================================================================================
# LinearPatternFormationNetwork

class LinearPatternFormationNetwork(torch.nn.Module):
  """
  Converts observations into actions with a single linear layer.

  There is one module:
  1. Select: `select_layer -> select_activation ->`

  `select_layer` is a linear layer that maps observations to actions, with weights and biases shared
  between segments according to their lateral, sagittal, and transverse relations.
  """
  def __init__(
    self,
    observation_prefixes: T.Mapping[str, T.Sequence[str]],
    action_segments: T.Sequence[str],
    action_prefixes: T.Sequence[str],

    lateral_segments: T.Mapping[str, tuple[str, str]] | None = None,
    sagittal_segments: T.Mapping[str, str] | None = None,
    transverse_segments: T.Mapping[str, str] | None = None,

    observation_prefix_shorthand: T.Mapping[str, str] | None = None,
    action_prefix_shorthand: T.Mapping[str, str] | None = None,

    action_prefix_symmetry_mapping: T.Mapping[str, str] | None = None,
    action_symmetry_mapping: T.Mapping[str, str] | None = None,

    select_weight_id_override: T.Mapping[str, str] | None = None,

    select_activation: T.Callable[[], torch.nn.Module] | None = None,

    on_initialize_select: T.Callable[[utb.BlockModule], None] | None = None,
  ):
    """
    Args:
      observation_prefixes: Mapping from body segment to the prefixes of its observations,
        e.g. `{'limbFL': ['rg', 'body/proprio_pos'], 'lumbarE': ['body/proprio_pos']}`, giving
        observation keys f'{prefix}_{segment}', enumerated prefix-first in order of appearance.
        Lateral pairs must observe the same prefixes.
      action_segments: Body segments with which actions may be associated,
        e.g. `['limbFL', 'limbFR', 'limbHL', 'limbHR', 'lumbarE', 'lumbarBT']`.
      action_prefixes: List of action prefixes, e.g. `['body/muscle_flx', 'body/muscle_ext']`.

      lateral_segments: Mapping from contralateral symmetry group to the left and right segments,
        e.g. {F: ['limbFL', 'limbFR']}.
      sagittal_segments: Mapping from sagittal symmetry group to the segment,
        e.g. {Le: 'lumbarE'}.
      transverse_segments: Mapping from transverse symmetry group to the segment,
        e.g. {Lbt: 'lumbarBT'}.

      observation_prefix_shorthand: Mapping from observation prefix to shorthand,
        e.g. `{'rg': 'rg', 'body/proprio_pos': 'pos'}`.
      action_prefix_shorthand: Mapping from action prefix to shorthand,
        e.g. `{'body/muscle_flx': 'flx'}`.

      action_prefix_symmetry_mapping: Mapping from action prefix to symmetry group,
        e.g. `{'body/muscle_flx': 'flx'}`.
      action_symmetry_mapping: Mapping from action key to symmetry group,
        e.g. `{'body/muscle_flx_FL': 'flx_F'}`.

      select_weight_id_override: Mapping from select weight ID to override weight ID.

      select_activation: Activation function for the select layer.

      on_initialize_select: Initializer for the select layer weights and biases.
    """
    super().__init__()
    assert all(len(prefixes) == len(set(prefixes)) for prefixes in observation_prefixes.values())
    assert len(action_segments) == len(set(action_segments))
    assert len(action_prefixes) == len(set(action_prefixes))
    segments = set(observation_prefixes) | set(action_segments)

    assert not lateral_segments or all(
      all(segment in segments for segment in pair) for pair in lateral_segments.values()
    )
    assert not sagittal_segments or all(
      segment in segments for segment in sagittal_segments.values()
    )
    assert not transverse_segments or all(
      segment in segments for segment in transverse_segments.values()
    )

    observation_prefix_shorthand = observation_prefix_shorthand or {}
    action_prefix_shorthand = action_prefix_shorthand or {}
    lateral_segments = lateral_segments or {}
    sagittal_segments = sagittal_segments or {}
    transverse_segments = transverse_segments or {}
    action_prefix_symmetry_mapping = action_prefix_symmetry_mapping or {}
    action_symmetry_mapping = action_symmetry_mapping or {}
    select_weight_id_override = select_weight_id_override or {}

    # Observation and action keys are expected to have the format f'{prefix}_{segment}',
    # e.g. body/proprio_pos_FL and body/muscle_flx_FL.
    # We enumerate all prefix and segment combinations to get concrete keys.
    # Enumerate observations prefix-first, e.g. all `rg` keys, then all `body/proprio_pos` keys.
    prefixes = list(dict.fromkeys(
      prefix for segment_prefixes in observation_prefixes.values() for prefix in segment_prefixes
    ))
    observation_prefix_segments = [
      (prefix, segment)
      for prefix in prefixes
      for segment, segment_prefixes in observation_prefixes.items()
      if prefix in segment_prefixes
    ]
    self.observation_keys = [
      f'{prefix}_{segment}' for prefix, segment in observation_prefix_segments
    ]
    # Lateral pairs share weights, so both sides must observe the same prefixes.
    for left, right in lateral_segments.values():
      assert set(observation_prefixes.get(left, [])) == set(observation_prefixes.get(right, [])), (
        f'Lateral segments {left} and {right} must observe the same prefixes: '
        f'{observation_prefixes.get(left)} vs {observation_prefixes.get(right)}.'
      )
    self.action_keys = [
      f'{prefix}_{suffix}' for prefix in action_prefixes for suffix in action_segments
    ]

    # Prevent bugs where the user thinks they have overridden something they have not.
    def assert_matches(name: str, default: T.Collection[str], override: T.Collection[str]):
      unmatched = [o for o in override if o not in default]
      assert not unmatched, (
        f'Overrides provided for {name}: {unmatched} do not correspond to anything: {default}.'
      )
    assert_matches('observation prefixes', prefixes, observation_prefix_shorthand)
    assert_matches('action prefixes', action_prefixes, action_prefix_shorthand)
    assert_matches('action prefixes', action_prefixes, action_prefix_symmetry_mapping)
    assert_matches('action keys', self.action_keys, action_symmetry_mapping)

    bodyplan = BodyPlan(lateral_segments, sagittal_segments, transverse_segments)

    # Define shared biases for actions.
    # Preferentially use full action_symmetry_mapping over individual prefix/suffix mappings.
    # Action bias IDs do not allow prefix symmetry because then different action prefixes would
    # be fully redundant. Same input, same weight, and same bias.
    select_bias_ids = [
      action_symmetry_mapping.get(
        f'{prefix}_{segment}',
        (
          action_prefix_symmetry_mapping.get(prefix, action_prefix_shorthand.get(prefix, prefix))
          + f'_{bodyplan.classify_segment(segment)[0]}'
        )
      )
      for prefix in action_prefixes
      for segment in action_segments
    ]

    # Define shared weights for actions.
    select_weight_ids = [
      [
        # e.g. posflxF>Fc, velextLe>Lbtc
        f'{observation_prefix_shorthand.get(observation_prefix, observation_prefix)}'
        f'{action_prefix_shorthand.get(action_prefix, action_prefix)}'
        f'{bodyplan.intersegment_weight_id(observation_segment, action_segment)}'
        for observation_prefix, observation_segment in observation_prefix_segments
      ]
      for action_prefix in action_prefixes
      for action_segment in action_segments
    ]
    select_weight_ids = [
      [
        select_weight_id_override.get(weight_id, weight_id) for weight_id in row
      ]
      for row in select_weight_ids
    ]

    # Select layer.
    self.select_layer = utb.BlockLinear(
      input_keys=self.observation_keys,
      output_keys=self.action_keys,
      weight_ids=select_weight_ids,
      bias_ids=select_bias_ids,
      on_initialize=on_initialize_select,
    )
    self.select_activation = (
      DictModule(select_activation, self.action_keys) if select_activation else None
    )

  def initialize(
    self,
    observation_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Dict,
  ):
    self.select_layer.initialize(observation_space, action_space)

  def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # 1. Compute actions. `a = o @ W + b`
    action = self.select_layer(observation)
    if self.select_activation:
      action = self.select_activation(action)
    return action


# ==================================================================================================
# SynergyPatternFormationNetwork

class SynergyPatternFormationNetwork(torch.nn.Module):
  """
  Converts observations into actions through a layer of muscle synergies.

  There are two sequential modules:
  1. Select: `select_layer -> select_activation ->`
  2. Synergy: `synergy_layer -> synergy_activation ->`

  `select_layer` is a linear layer that maps observations to one activation per entry in
  `segment_synergies[synergy_segment]`, for each synergy segment. Its weights and biases are shared
  between segments according to their lateral, sagittal, and transverse relations.

  `synergy_layer` is a sparse linear layer that maps each action from the synergy activations of
  the segments listed for its action segment in `action_synergies`. Each synergy is a set of
  per-action-prefix column weights, named in `synergies`. A synergy may appear in multiple segments
  (e.g. `limbFL` and `limbFR`), or multiple times in one segment, giving it multiple independently
  weighted activation routes.

  The network is unbatched: every observation and action is one sample's vector.

  Parameter IDs (shared parameters have the same ID, and the IDs are the names that initializers
  and `select_weight_id_override` refer to):
  - Select weights: `{observation shorthand}{BodyPlan.intersegment_weight_id(observation segment,
    synergy segment)}`, e.g. `posF>F` (a forelimb's positions onto its own synergies) or
    `rgF>H` (a forelimb's RG outputs onto a hindlimb's synergies). The block of each (observation
    key, synergy segment) pair has shape `(synergy count, observation size)`.
  - Select biases: the synergy segment's symmetry group, e.g. `F`, `H`, `Le`, `Lbt`.
  - Synergy weights: the weight IDs in `synergies`, one `(action size, 1)` column per synergy and
    action prefix, shared by every segment and route that lists the synergy. `-(...)` negates the
    shared column (e.g. the mirrored lumbar synergies `Lbt_1c: '-(Lbt_1_flx)'`).
  - Synergy (action) biases: `action_symmetry_mapping[action key]` if given, else
    `{action_prefix_symmetry_mapping or action_prefix_shorthand or prefix}_{segment group}`,
    e.g. `act_F`.

  For example, with the flexor-extensor config (`pattern_formation/limbs28_lumbar3/flxext`), each
  limb segment computes 13 select activations (one per route) from its RG, proprioceptive and
  tactile observations, and each `body/muscle_{flx,ext}_<limb>` action (one value per limb muscle)
  sums the columns of the limb group's 5 synergies of its prefix, each weighted by the activations
  of the synergy's routes. Per interface, the action keys are
  `body/actuators_trq_<segment>` (torque), `body/actuators_pos_<segment>` (position, and
  `body/actuators_kp_<segment>` for position-stiffness), or `body/muscle_{flx,ext}_<segment>`
  (flexor-extensor).
  """
  def __init__(
    self,
    observation_prefixes: T.Mapping[str, T.Sequence[str]],
    action_segments: T.Sequence[str],
    action_prefixes: T.Sequence[str],

    synergies: T.Mapping[str, T.Mapping[str, utb.WeightId | T.Literal[0]] | None],
    segment_synergies: T.Mapping[str, T.Sequence[str]],
    action_synergies: T.Mapping[str, T.Sequence[str]],

    lateral_segments: T.Mapping[str, tuple[str, str]] | None = None,
    sagittal_segments: T.Mapping[str, str] | None = None,
    transverse_segments: T.Mapping[str, str] | None = None,

    observation_prefix_shorthand: T.Mapping[str, str] | None = None,
    action_prefix_shorthand: T.Mapping[str, str] | None = None,

    action_prefix_symmetry_mapping: T.Mapping[str, str] | None = None,
    action_symmetry_mapping: T.Mapping[str, str] | None = None,

    select_weight_id_override: T.Mapping[str, str] | None = None,

    select_activation: T.Callable[[], torch.nn.Module] | None = None,
    synergy_activation: T.Callable[[], torch.nn.Module] | None = torch.nn.ReLU,

    on_initialize_select: T.Callable[[utb.BlockModule], None] | None = None,
    on_initialize_synergy: T.Callable[[utb.BlockModule], None] | None = None,
  ):
    """Initializes the network.

    Args:
      observation_prefixes: Mapping from body segment to the prefixes of its observations,
        e.g. `{'limbFL': ['rg', 'body/proprio_pos'], 'lumbarE': ['body/proprio_pos']}`, giving
        observation keys f'{prefix}_{segment}', enumerated prefix-first in order of appearance.
        Lateral pairs must observe the same prefixes.
      action_segments: Body segments with which actions may be associated,
        e.g. `['limbFL', 'limbFR', 'limbHL', 'limbHR', 'lumbarE', 'lumbarBT']`.
      action_prefixes: List of action prefixes, e.g. `['body/muscle_flx', 'body/muscle_ext']`.

      synergies: Mapping from synergy name to the synergy layer weight ID for each action prefix,
        e.g. `{'Lbt_1c': {'body/muscle_flx': '-(Lbt_1_flx)', 'body/muscle_ext': '-(Lbt_1_ext)'}}`.
        Synergies mapped to `None` are ignored (so config variants can drop inherited ones).
        Every synergy must specify every action prefix; `0` disables a prefix. Each weight ID is a
        `(segment action size, 1)` column, and may include a weight transform such as `-(...)`.
      segment_synergies: Mapping from synergy segment to the synergies it activates,
        e.g. `{'limbFL': ['F_flx_1', 'F_flx_2'], 'lumbarBT[->]': ['Lbt_1c']}`. Repeated synergies
        are labeled `name#1`, `name#2`, ... in the select layer. Lateral pairs must list synergies
        whose weight IDs match up to transforms, in the same order.
      action_synergies: Mapping from action segment to the synergy segments that contribute to it,
        e.g. `{'limbFL': ['limbFL'], 'lumbarBT': ['lumbarBT[<-]', 'lumbarBT[->]']}`.

      lateral_segments: Mapping from contralateral symmetry group to the left and right segments,
        e.g. {F: ['limbFL', 'limbFR']}.
      sagittal_segments: Mapping from sagittal symmetry group to the segment,
        e.g. {Le: 'lumbarE'}.
      transverse_segments: Mapping from transverse symmetry group to the segment,
        e.g. {Lbt: 'lumbarBT'}.

      observation_prefix_shorthand: Mapping from observation prefix to shorthand,
        e.g. `{'rg': 'rg', 'body/proprio_pos': 'pos'}`.
      action_prefix_shorthand: Mapping from action prefix to shorthand,
        e.g. `{'body/muscle_flx': 'flx'}`.

      action_prefix_symmetry_mapping: Mapping from action prefix to symmetry group,
        e.g. `{'body/muscle_flx': 'flx'}`.
      action_symmetry_mapping: Mapping from action key to symmetry group,
        e.g. `{'body/muscle_flx_FL': 'flx_F'}`.

      select_weight_id_override: Mapping from select weight ID to override weight ID, e.g. to make
        an intersegmental connection a frozen zero. An override that matches no ID is ignored.

      select_activation: Activation function for the select layer. The synergy activations are
        declared in [0, 1] for the synergy layer, whatever the activation.
      synergy_activation: Activation function for the synergy layer.

      on_initialize_select: Initializer for the select layer weights and biases.
      on_initialize_synergy: Initializer for the synergy layer weights and biases.
    """
    super().__init__()
    assert all(len(prefixes) == len(set(prefixes)) for prefixes in observation_prefixes.values())
    assert len(action_segments) == len(set(action_segments))
    assert len(action_prefixes) == len(set(action_prefixes))
    segments = set(observation_prefixes) | set(action_segments)

    # A `None` synergy is removed, so that config variants can drop inherited synergies.
    synergy_ids = {
      name: prefix_ids for name, prefix_ids in synergies.items() if prefix_ids is not None
    }
    assert all(segment in action_synergies for segment in action_segments)
    assert all(
      all(segment in segments for segment in synergy_segments)
      for synergy_segments in action_synergies.values()
    )
    synergy_segments = list(dict.fromkeys(
      segment for action_segment in action_segments for segment in action_synergies[action_segment]
    ))
    assert set(segment_synergies) == set(synergy_segments), (
      f'segment_synergies {list(segment_synergies)} must specify exactly the synergy segments '
      f'referenced by action_synergies {synergy_segments}.'
    )
    assert all(len(names) > 0 for names in segment_synergies.values())
    used_synergies = set(name for names in segment_synergies.values() for name in names)
    assert used_synergies == set(synergy_ids), (
      f'Undefined synergies: {used_synergies - set(synergy_ids)}. '
      f'Unused synergies: {set(synergy_ids) - used_synergies}.'
    )
    assert all(set(prefix_ids) == set(action_prefixes) for prefix_ids in synergy_ids.values()), (
      f'Every synergy must specify a weight ID for each action prefix {action_prefixes}.'
    )
    assert not lateral_segments or all(
      all(segment in segments for segment in pair) for pair in lateral_segments.values()
    )
    assert not sagittal_segments or all(
      segment in segments for segment in sagittal_segments.values()
    )
    assert not transverse_segments or all(
      segment in segments for segment in transverse_segments.values()
    )

    observation_prefix_shorthand = observation_prefix_shorthand or {}
    action_prefix_shorthand = action_prefix_shorthand or {}
    lateral_segments = lateral_segments or {}
    sagittal_segments = sagittal_segments or {}
    transverse_segments = transverse_segments or {}
    action_prefix_symmetry_mapping = action_prefix_symmetry_mapping or {}
    action_symmetry_mapping = action_symmetry_mapping or {}
    select_weight_id_override = select_weight_id_override or {}

    # Lateral pairs share select weights row-for-row, so their synergies must correspond by
    # position: the same weight IDs up to transforms (e.g. `pos_Lbt1` and `-(pos_Lbt1)`).
    def base_weight_ids(segment: str) -> list[list[utb.WeightId | T.Literal[0]]]:
      return [
        [
          utb.parse_weight_transform(weight_id)[0] if isinstance(weight_id, str) else weight_id
          for weight_id in (synergy_ids[name][prefix] for prefix in action_prefixes)
        ]
        for name in segment_synergies[segment]
      ]
    for left, right in lateral_segments.values():
      if left in segment_synergies and right in segment_synergies:
        assert base_weight_ids(left) == base_weight_ids(right), (
          f'Lateral segments {left} and {right} must list corresponding synergies in the same '
          f'order: {segment_synergies[left]} vs {segment_synergies[right]}.'
        )

    # Observation and action keys are expected to have the format f'{prefix}_{segment}',
    # e.g. body/proprio_pos_FL and body/muscle_flx_FL.
    # We enumerate all prefix and segment combinations to get concrete keys.
    # Enumerate observations prefix-first, e.g. all `rg` keys, then all `body/proprio_pos` keys.
    prefixes = list(dict.fromkeys(
      prefix for segment_prefixes in observation_prefixes.values() for prefix in segment_prefixes
    ))
    observation_prefix_segments = [
      (prefix, segment)
      for prefix in prefixes
      for segment, segment_prefixes in observation_prefixes.items()
      if prefix in segment_prefixes
    ]
    self.observation_keys = [
      f'{prefix}_{segment}' for prefix, segment in observation_prefix_segments
    ]
    # Lateral pairs share weights, so both sides must observe the same prefixes.
    for left, right in lateral_segments.values():
      assert set(observation_prefixes.get(left, [])) == set(observation_prefixes.get(right, [])), (
        f'Lateral segments {left} and {right} must observe the same prefixes: '
        f'{observation_prefixes.get(left)} vs {observation_prefixes.get(right)}.'
      )
    self.action_keys = [
      f'{prefix}_{suffix}' for prefix in action_prefixes for suffix in action_segments
    ]
    # Label each synergy activation by its synergy name, tagging repeats as `name#1`, `name#2`, ...
    self.synergy_labels: dict[str, list[str]] = {}
    for segment in synergy_segments:
      counts = collections.Counter()
      self.synergy_labels[segment] = []
      for name in segment_synergies[segment]:
        self.synergy_labels[segment].append(f'{name}#{counts[name]}' if counts[name] else name)
        counts[name] += 1

    # Prevent bugs where the user thinks they have overridden something they have not.
    def assert_matches(name: str, default: T.Collection[str], override: T.Collection[str]):
      unmatched = [o for o in override if o not in default]
      assert not unmatched, (
        f'Overrides provided for {name}: {unmatched} do not correspond to anything: {default}.'
      )
    assert_matches('observation prefixes', prefixes, observation_prefix_shorthand)
    assert_matches('action prefixes', action_prefixes, action_prefix_shorthand)
    assert_matches('action prefixes', action_prefixes, action_prefix_symmetry_mapping)
    assert_matches('action keys', self.action_keys, action_symmetry_mapping)

    bodyplan = BodyPlan(lateral_segments, sagittal_segments, transverse_segments)

    # Define shared biases for actions.
    # Preferentially use full action_symmetry_mapping over individual prefix/suffix mappings.
    # Action bias IDs do not allow prefix symmetry because then different action prefixes would
    # be fully redundant. Same input, same weight, and same bias.
    synergy_bias_ids = [
      action_symmetry_mapping.get(
        f'{prefix}_{segment}',
        (
          action_prefix_symmetry_mapping.get(prefix, action_prefix_shorthand.get(prefix, prefix))
          + f'_{bodyplan.classify_segment(segment)[0]}'
        )
      )
      for prefix in action_prefixes
      for segment in action_segments
    ]
    # Define shared weights for synergies. Each action key's block takes every synergy activation of
    # its synergy segments, split into one key per activation, e.g. `lumbarBT[<-]/0`.
    synergy_input_keys = [
      tuple(
        f'{synergy_segment}/{i}'
        for synergy_segment in action_synergies[action_segment]
        for i in range(len(segment_synergies[synergy_segment]))
      )
      for _ in action_prefixes
      for action_segment in action_segments
    ]
    synergy_weight_ids = [
      (
        tuple(
          synergy_ids[name][action_prefix]
          for synergy_segment in action_synergies[action_segment]
          for name in segment_synergies[synergy_segment]
        ),
      )
      for action_prefix in action_prefixes
      for action_segment in action_segments
    ]

    # Define shared weights and biases for synergy activations. Synergy keys have no prefix.
    select_bias_ids = [bodyplan.classify_segment(segment)[0] for segment in synergy_segments]
    select_weight_ids = [
      [
        # e.g. posF>Fc, velLe>Lbtc
        f'{observation_prefix_shorthand.get(observation_prefix, observation_prefix)}'
        f'{bodyplan.intersegment_weight_id(observation_segment, synergy_segment)}'
        for observation_prefix, observation_segment in observation_prefix_segments
      ]
      for synergy_segment in synergy_segments
    ]
    select_weight_ids = [
      [
        select_weight_id_override.get(weight_id, weight_id) for weight_id in row
      ]
      for row in select_weight_ids
    ]

    # Select layer.
    self.select_layer = utb.BlockLinear(
      input_keys=self.observation_keys,
      output_keys=synergy_segments,
      weight_ids=select_weight_ids,
      bias_ids=select_bias_ids,
      on_initialize=on_initialize_select,
    )
    self.select_activation = (
      DictModule(select_activation, synergy_segments) if select_activation else None
    )

    # Synergy layer.
    self.synergy_layer = utb.SparseBlockLinear(
      input_keys=synergy_input_keys,
      output_keys=list(zip(self.action_keys)),
      weight_ids=synergy_weight_ids,
      bias_ids=list(zip(synergy_bias_ids)),
      on_initialize=on_initialize_synergy,
    )
    self.synergy_activation = (
      DictModule(synergy_activation, self.action_keys) if synergy_activation else None
    )

  def initialize(
    self,
    observation_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Dict,
  ):
    """Initializes the select and synergy layers for the spaces.

    Args:
      observation_space: The PF's observation space (see `RatNcapActor`).
      action_space: Action space, with every key of `action_prefixes` x `action_segments`.
    """
    # The select layer outputs one vector per synergy segment, which is split by index into one key
    # per synergy activation for the synergy layer.
    select_space = gymnasium.spaces.Dict({
      segment: cxg.Box(low=0.0, high=1.0, shape=(len(labels),), labels=[labels])
      for segment, labels in self.synergy_labels.items()
    })
    synergy_space = gymnasium.spaces.Dict({
      f'{segment}/{i}': cxg.Box(low=0.0, high=1.0, shape=(1,), labels=[[label]])
      for segment, labels in self.synergy_labels.items()
      for i, label in enumerate(labels)
    })
    self.select_layer.initialize(observation_space, select_space)
    self.synergy_layer.initialize(synergy_space, action_space)

  def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Returns the actions of one sample, {action key: [size]}, from its observations."""
    # 1. Compute synergy activations. `s = o @ W + b`
    synergy = self.select_layer(observation)
    if self.select_activation:
      synergy = self.select_activation(synergy)
    # 2. Split each synergy segment's activations by index.
    synergy = {
      f'{segment}/{i}': activations[i:i + 1]
      for segment, activations in synergy.items()
      for i in range(activations.shape[0])
    }
    # 3. Compute actions from synergy activations. `a_i = s_i @ W_i + b_i`
    action = self.synergy_layer(synergy)
    if self.synergy_activation:
      action = self.synergy_activation(action)
    return action


# ==================================================================================================
# AfferentFeedback

class AfferentFeedbackNetwork(torch.nn.Module):
  """
  Converts observations into sensor (non-resetting) and resetting afferent feedback signals.

  There are two sequential modules:
  1. Sensor: `sensor_layer -> sensor_activation ->`
  2. Reset: `reset_layer -> reset_activation ->`

  `sensor_layer` is a scalar weight and bias that converts observations into sensor signals.
  `reset_layer` is a linear layer that converts sensor signals into resetting afferent signals.

  In `RatNcapActor`, the sensor signals are the PF's observations (so the PF observes every prefix
  through the AF, except the RG and BC outputs), and the reset signals drive the RG's afferent
  signals: the action prefixes are `flx` and `ext`, and the action segments are the four limbs
  (`limbFL`, ...).

  The network is unbatched: every observation and output is one sample's vector.

  Parameter IDs: sensor weights and biases are `{observation prefix}_{segment group}` (e.g.
  `body/proprio_pos_F`); reset weights are `{observation shorthand}{action shorthand}`
  `{BodyPlan.intersegment_weight_id(observation segment, action segment)}` (e.g. `posflxF>F`);
  reset biases are `action_symmetry_mapping[action key]` if given, else
  `{action_prefix_symmetry_mapping or prefix}_{segment group}`.
  """
  def __init__(
    self,
    observation_prefixes: T.Mapping[str, T.Sequence[str]],
    action_segments: T.Sequence[str],
    action_prefixes: T.Sequence[str],

    lateral_segments: T.Mapping[str, tuple[str, str]] | None = None,
    sagittal_segments: T.Mapping[str, str] | None = None,
    transverse_segments: T.Mapping[str, str] | None = None,

    observation_prefix_shorthand: T.Mapping[str, str] | None = None,
    action_prefix_shorthand: T.Mapping[str, str] | None = None,

    action_prefix_symmetry_mapping: T.Mapping[str, str] | None = None,
    action_symmetry_mapping: T.Mapping[str, str] | None = None,

    reset: bool = False,

    reset_weight_id_override: T.Mapping[str, str] | None = None,

    sensor_activation: T.Callable[[], torch.nn.Module] | None = None,
    reset_activation: T.Callable[[], torch.nn.Module] | None = None,

    on_initialize_sensor: T.Callable[[utb.BlockModule], None] | None = None,
    on_initialize_reset: T.Callable[[utb.BlockModule], None] | None = None,
  ):
    """Initializes the network.

    Args:
      observation_prefixes: Mapping from body segment to the prefixes of its observations,
        e.g. `{'limbFL': ['rg', 'body/proprio_pos'], 'lumbarE': ['body/proprio_pos']}`, giving
        observation keys f'{prefix}_{segment}', enumerated prefix-first in order of appearance.
        Lateral pairs must observe the same prefixes.
      action_segments: Body segments with which actions may be associated, e.g.
        `['limbFL', 'limbFR', 'limbHL', 'limbHR', 'lumbarE', 'lumbarBT']`.
      action_prefixes: List of action prefixes, e.g. `['body/muscle_flx', 'body/muscle_ext']`.

      lateral_segments: Mapping from contralateral symmetry group to the left and right segments,
        e.g. {F: ['limbFL', 'limbFR']}.
      sagittal_segments: Mapping from sagittal symmetry group to the segment,
        e.g. {Le: 'lumbarE'}.
      transverse_segments: Mapping from transverse symmetry group to the segment,
        e.g. {Lbt: 'lumbarBT'}.

      observation_prefix_shorthand: Mapping from observation prefix to shorthand, e.g.
        `{'rg': 'rg', 'body/proprio_pos': 'pos'}`.
      action_prefix_shorthand: Mapping from action prefix to shorthand, e.g.
        `{'body/muscle_flx': 'flx'}`.

      action_prefix_symmetry_mapping: Mapping from action prefix to symmetry group, e.g.
        `{'body/muscle_flx': 'flx'}`.
      action_symmetry_mapping: Mapping from action key to symmetry group, e.g.
        `{'body/muscle_flx_FL': 'flx_F'}`.

      reset: Whether to compute the reset signals. The reset layer is built either way.

      reset_weight_id_override: Mapping from reset weight ID to override weight ID.

      sensor_activation: Activation function for the sensory layer.
      reset_activation: Activation function for the reset layer.

      on_initialize_sensor: Initializer for the sensory layer weights and biases.
      on_initialize_reset: Initializer for the reset layer weights and biases.
    """
    super().__init__()
    assert all(len(prefixes) == len(set(prefixes)) for prefixes in observation_prefixes.values())
    assert len(action_segments) == len(set(action_segments))
    assert len(action_prefixes) == len(set(action_prefixes))

    assert reset or not reset_activation
    assert reset or not on_initialize_reset

    self.reset = reset

    observation_prefix_shorthand = observation_prefix_shorthand or {}
    action_prefix_shorthand = action_prefix_shorthand or {}
    lateral_segments = lateral_segments or {}
    sagittal_segments = sagittal_segments or {}
    transverse_segments = transverse_segments or {}
    action_prefix_symmetry_mapping = action_prefix_symmetry_mapping or {}
    action_symmetry_mapping = action_symmetry_mapping or {}
    reset_weight_id_override = reset_weight_id_override or {}

    bodyplan = BodyPlan(lateral_segments, sagittal_segments, transverse_segments)

    # Observation and action keys are expected to have the format f'{prefix}_{segment}',
    # e.g. body/proprio_pos_FL and body/muscle_flx_FL.
    # We enumerate all prefix and segment combinations to get concrete keys.
    # Enumerate observations prefix-first, e.g. all `rg` keys, then all `body/proprio_pos` keys.
    prefixes = list(dict.fromkeys(
      prefix for segment_prefixes in observation_prefixes.values() for prefix in segment_prefixes
    ))
    observation_prefix_segments = [
      (prefix, segment)
      for prefix in prefixes
      for segment, segment_prefixes in observation_prefixes.items()
      if prefix in segment_prefixes
    ]
    self.observation_keys = [
      f'{prefix}_{segment}' for prefix, segment in observation_prefix_segments
    ]
    # Lateral pairs share weights, so both sides must observe the same prefixes.
    for left, right in lateral_segments.values():
      assert set(observation_prefixes.get(left, [])) == set(observation_prefixes.get(right, [])), (
        f'Lateral segments {left} and {right} must observe the same prefixes: '
        f'{observation_prefixes.get(left)} vs {observation_prefixes.get(right)}.'
      )
    self.action_keys = [
      f'{prefix}_{suffix}' for prefix in action_prefixes for suffix in action_segments
    ]

    # Define shared weights for sensory layer.
    sensor_ids = [
      f'{prefix}_{bodyplan.classify_segment(segment)[0]}'
      for prefix, segment in observation_prefix_segments
    ]

    # Define shared weights for reset layer.
    # Preferentially use full action_symmetry_mapping over individual prefix/suffix mappings.
    reset_weight_ids = [
      [
        # e.g. posF>Fc, velLe>Lbtc
        f'{observation_prefix_shorthand.get(observation_prefix, observation_prefix)}'
        f'{action_prefix_shorthand.get(action_prefix, action_prefix)}'
        f'{bodyplan.intersegment_weight_id(observation_segment, action_segment)}'
        for observation_prefix, observation_segment in observation_prefix_segments
      ]
      for action_prefix in action_prefixes
      for action_segment in action_segments
    ]
    reset_weight_ids = [
      [reset_weight_id_override.get(weight_id, weight_id) for weight_id in row]
      for row in reset_weight_ids
    ]
    reset_bias_ids = [
      action_symmetry_mapping.get(
        f'{prefix}_{segment}',
        (
          f'{action_prefix_symmetry_mapping.get(prefix, prefix)}_'
          f'{bodyplan.classify_segment(segment)[0]}'
        )
      )
      for prefix in action_prefixes
      for segment in action_segments
    ]

    # Sensor layer.
    self.sensor_layer = utb.BlockScaleShift(
      input_keys=self.observation_keys,
      weight_ids=sensor_ids,
      bias_ids=sensor_ids,
      on_initialize=on_initialize_sensor,
    )
    self.sensor_activation = (
      DictModule(sensor_activation, self.observation_keys) if sensor_activation else None
    )

    # Reset layer.
    self.reset_layer = utb.BlockLinear(
      input_keys=self.observation_keys,
      output_keys=self.action_keys,
      weight_ids=reset_weight_ids,
      bias_ids=reset_bias_ids,
      on_initialize=on_initialize_reset,
    )

    self.reset_activation = (
      DictModule(reset_activation, self.action_keys) if reset_activation else None
    )

  def initialize(
    self,
    observation_space: gymnasium.spaces.Dict,
    action_space: gymnasium.spaces.Dict,
  ):
    """Initializes the sensor layer, and the reset layer if `reset`, for the spaces.

    Args:
      observation_space: Observation space (in `RatNcapActor`, the rectified observations and the
        RG outputs).
      action_space: Space of the reset signals.
    """
    self.sensor_layer.initialize(observation_space)
    if self.reset:
      self.reset_layer.initialize(observation_space, action_space)

  def forward(
    self,
    observation: dict[str, torch.Tensor]
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Returns the sensor signals {observation key: [size]} and the reset signals
    {action key: [size]} (`None` without `reset`) of one sample."""
    # 1. Scale each observation key and add a bias. `o_i = o_i * w_i + b_i`
    sensor = self.sensor_layer(observation)
    if self.sensor_activation:
      sensor = self.sensor_activation(sensor)

    # 2. Compute reset. `a = o @ W + b`
    if self.reset:
      reset = self.reset_layer(sensor)
      if self.reset_activation:
        reset = self.reset_activation(reset)
    else:
      reset = None

    return sensor, reset


# The type of a segment (see `BodyPlan`). 'saggital' (sic) is kept as spelled: it is a value, not
# an argument name.
SegmentType = T.Literal['lateral-left', 'lateral-right', 'saggital', 'transverse']

class BodyPlan:
  """
  Utility class for determining intersegmental relationships.

  We use this to set the shared weights and symmetries.
  This produces a 2D list filled with unique weight block identifiers
  f'{obs}{act}{obs_segment}->{act_segment}' like:

  'rgfF>F' (rg: RG output, f: flexor muscle abbrev, F->F: front limb to same front limb)
    - The weight matrix that defines the contribution of RG output from a front limb's flexor and
      extensor units to that SAME front limb's flexor muscle action, shared between FL and FR.

  'rgeF>Fc' (rg: RG output, e: extensor muscle abbrev, F->Fc: front limb to cross front limb)
    - The weight matrix that defines the contribution of RG output from a front limb's flexor and
      extensor units to the OTHER front limb's extensor muscle action, shared between FL and FR.

  'posfLe>H' (pos: body/proprio_pos, f: flexor muscle abbrev, Le: lumbar_extend, H: hind limb)
    - The weight matrix that defines the contribution of the lumbar_extend position observation
      to the hind limb's flexor muscle action, shared between both hind limbs.

  These weight block identifiers can also be surrounded with '-(...)' to indicate a sign-flipped
  relation, important for asymmetric observations like tendon bend and twist.
  When synergies are enabled, {act}={''}, because then individual action keys are delegated
  to the synergy layer, and the BlockLinear just controls `n` synergies per limb.

  Segments are grouped by the keys of the three mappings, and classified as:
  - lateral (left or right): one of a mirrored left/right pair, e.g. the forelimbs `F`;
  - sagittal: a midline segment that is mirror-symmetric, e.g. lumbar extension `Le`;
  - transverse: a midline segment that is mirror-antisymmetric, e.g. lumbar bend and twist `Lbt`.
  A connection ID gets the `c` (contralateral) suffix between the left and right segments of
  lateral pairs, and between a transverse segment and a right lateral segment (not a left one):
  mirroring left and right flips the sign of a transverse segment's motion, which this asymmetry
  encodes for the lumbar segments.
  """
  def __init__(
    self,
    lateral_segments: T.Mapping[str, tuple[str, str]],
    sagittal_segments: T.Mapping[str, str],
    transverse_segments: T.Mapping[str, str]
  ):
    """Initializes the body plan from the mappings of symmetry group to segment(s)."""
    self.lateral_segments = lateral_segments
    self.sagittal_segments = sagittal_segments
    self.transverse_segments = transverse_segments

  def classify_segment(self, segment: str) -> tuple[str, SegmentType]:
    """Returns the symmetry group and type of `segment` (lateral pairs first, then sagittal, then
    transverse).

    Raises:
      ValueError: If `segment` is in none of the mappings.
    """
    for contralateral_group, (left, right) in self.lateral_segments.items():
      if segment == left:
        return contralateral_group, 'lateral-left'
      elif segment == right:
        return contralateral_group, 'lateral-right'
    for sagittal_group, sagittal_segment in self.sagittal_segments.items():
      if segment == sagittal_segment:
        return sagittal_group, 'saggital'
    for transverse_group, transverse_segment in self.transverse_segments.items():
      if segment == transverse_segment:
        return transverse_group, 'transverse'
    raise ValueError(f'Segment {segment} is not classified as lateral, saggital, or transverse.')

  def intersegment_weight_id(self, segment1: str, segment2: str) -> str:
    """Returns the ID `{group1}>{group2}` of the connection from `segment1` to `segment2`, with
    the `c` suffix if they are contralateral (see the class), e.g. `F>Fc` from `limbFL` to
    `limbFR`."""
    segment_group1, segment_type1 = self.classify_segment(segment1)
    segment_group2, segment_type2 = self.classify_segment(segment2)
    contralateral_pairs = [('lateral-left', 'lateral-right'), ('transverse', 'lateral-right')]
    contralateral = any(
      (segment_type1, segment_type2) == pair or (segment_type2, segment_type1) == pair
      for pair in contralateral_pairs
    )
    return f'{segment_group1}>{segment_group2}{"c" if contralateral else ""}'


# ==================================================================================================
# Brainstem Command modules

def init_bc_fixed(
  head: torch.nn.Module,
  torso: torch.nn.Module | None = None,
  weight: float | list = 0.5,
  bias: float | list = -0.05,
  weight_grad: bool = False,
  bias_grad: bool = False,
):
  """Sets the BC head's linear layer to a fixed weight and bias (`BrainstemCommandNetwork.init`).

  With the provided config values (weight 1, bias 0, both frozen), the `speed` command equals the
  `target_vx` observation.

  Args:
    head: The BC head, a tonic `DeterministicPolicyHead`.
    torso: The BC torso (unused).
    weight: Weight value(s) of the head's linear layer.
    bias: Bias value(s) of the head's linear layer.
    weight_grad: Whether the weight is trained.
    bias_grad: Whether the bias is trained.
  """
  head = T.cast(torch.nn.Linear, head.action_layer[0])  # type: ignore
  head.weight.data.copy_(torch.tensor(weight))
  head.weight.requires_grad_(weight_grad)
  if head.bias is not None:
    head.bias.data.copy_(torch.tensor(bias))
    head.bias.requires_grad_(bias_grad)


class BrainstemCommandSignal(torch.nn.Module):
  """A Brainstem Command (BC) signal module generates fixed named command signals, e.g. `speed`, to
  control the RG and PF."""
  def __init__(
    self,
    command: T.Mapping[str, float],
    grad: bool = True,
    repeat: int = 1,
    repeat_agg: T.Literal['mean', 'sum'] = 'mean',
  ):
    """
    Args:
      command: Mapping from command name to its fixed value, e.g. `{'speed': 0.5}`.
    """
    super().__init__()
    self.commands = list(command)
    values = torch.tensor([float(v) for v in command.values()], dtype=torch.float32)
    values = values.unsqueeze(1).repeat_interleave(repeat, dim=-1) / repeat
    self.command = torch.nn.Parameter(values, requires_grad=grad)  # [len(commands), repeat]
    self.repeat = repeat

    match repeat_agg:
      case 'mean':
        self.repeat_agg = functools.partial(torch.mean, dim=-1, keepdim=False)
      case 'sum':
        self.repeat_agg = functools.partial(torch.sum, dim=-1, keepdim=False)
      case _:
        raise ValueError(f'Invalid repeat_agg: {repeat_agg}')

  def initialize(self, observation_space: gymnasium.spaces.Dict) -> gymnasium.spaces.Dict:
    """Returns the command space, e.g. `{'speed': Box(shape=(1,))}`."""
    return gymnasium.spaces.Dict([
      (command, cxg.Box(low=-1.0, high=1.0, shape=(1,), labels=[[command]]))
      for command in self.commands
    ])

  def forward(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Args:
      observations: Dict observations with tensor values [observation_value_size].
    Return:
      commands: Dict commands with tensor values [1].
    """
    command = self.repeat_agg(self.command)  # [len(commands)]
    return {name: command[i:i + 1] for i, name in enumerate(self.commands)}


class BrainstemCommandNetwork(torch.nn.Module):
  """A Brainstem Command (BC) network module generates variable named command signals, e.g. `speed`,
  to control the RG and PF."""
  def __init__(
    self,
    torso: torch.nn.Module | None = None,
    head: torch.nn.Module | None = None,
    observe: T.Sequence[str] = ('target_vx',),
    repeat: int = 1,
    commands: T.Sequence[str] = ('speed',),
    init: T.Callable[..., None] = init_bc_fixed,
    passthrough: T.Sequence[str] = (),
  ):
    """Initializes the network.

    Args:
      torso: Network between the observations and the head, or `None` for none.
      head: Linear head with one output per command. Defaults to a zero-initialized linear layer.
      observe: Observation keys that the network takes, concatenated in the observations' order
        (not this list's).
      repeat: Number of times each observation value is repeated before the torso.
      commands: Names of the commands, one head output each, e.g. `['speed']`.
      passthrough: Names of 1D observations that are passed through unmodified as additional
        commands of the same name, e.g. `['prepare', 'launch']`. They bypass the torso and head.
      init: Function that initializes the head and torso, called as `init(head=..., torso=...)`.
    """
    super().__init__()
    assert not set(commands) & set(passthrough), (
      f'Commands {list(commands)} and passthrough commands {list(passthrough)} must be distinct.'
    )
    self.torso = torso
    self.head = head or ttm.DeterministicPolicyHead(
      activation=torch.nn.Identity,
      bias=True,
      init_fn=init_zeros,
    )
    self.observe = observe
    self.repeat = repeat
    self.commands = list(commands)
    self.init = init
    self.passthrough = list(passthrough)

  def initialize(self, observation_space: gymnasium.spaces.Dict) -> gymnasium.spaces.Dict:
    """Returns the command space, e.g. `{'speed': Box(shape=(1,))}`."""
    self.observation_space = observation_space
    observation_size = sum(
      T.cast(gymnasium.spaces.Box, observation_space[oname]).shape[0]
      for oname in self.observe
    )
    size = observation_size * self.repeat
    if self.torso is not None:
      size = self.torso.initialize(size)  # type: ignore
    self.head.initialize(size, gymnasium.spaces.Box(-1, 1, (len(self.commands),)))  # type: ignore
    self.init(head=self.head, torso=self.torso)
    command_space = gymnasium.spaces.Dict([
      (command, cxg.Box(low=-1.0, high=1.0, shape=(1,), labels=[[command]]))
      for command in self.commands
    ])
    # Passthrough commands keep the bounds of their observations.
    for command in self.passthrough:
      space = T.cast(gymnasium.spaces.Box, observation_space[command])
      assert space.shape == (1,), f'Passthrough command {command!r} must be 1D, got {space.shape}.'
      command_space[command] = cxg.Box(
        low=space.low, high=space.high, shape=(1,), labels=[[command]],
      )
    return command_space

  def forward(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Computes the commands of one sample.

    Args:
      observations: Observations of one sample, {key: [size]}.

    Returns:
      Commands of one sample, {command: [1]}.
    """
    out = torch.concat([o for oname, o in observations.items() if oname in self.observe], dim=-1)
    out = out.repeat_interleave(self.repeat, dim=-1)
    if self.torso is not None:
      out = self.torso(out)
    command = self.head(out)  # [len(commands)]
    commands = {name: command[i:i + 1] for i, name in enumerate(self.commands)}
    commands.update({name: observations[name] for name in self.passthrough})
    return commands


# ==================================================================================================
# ObservationRectifier

ObsKey = str
"""An observation key, e.g. `body/proprio_pos_limbFL`."""
ActKey = str
"""An action key, e.g. `body/muscle_flx_limbFL`."""
ConstrainFn = T.Callable[[torch.nn.Parameter | torch.Tensor], None]
"""A function that constrains a parameter (e.g. `constrain` with bound arguments)."""

RectifyMode = T.Literal['sign', 'flip', 'activation']
# Suffixes appended (as f'{oname}[{suffix}]') per split component of a rectification, in order.
RECTIFY_SUFFIX: dict[RectifyMode, tuple[str, ...]] = {
  'sign': ('-', '+'),    # x -> [ReLU(-x), ReLU(x)]
  'flip': ('<-', '->'),  # x -> [high - x + low, x]: x mirrored within its box, and x.
  'activation': ('',),
}


class ObservationRectifier(torch.nn.Module):
  """Splits/transforms observations by key, keyed by glob pattern (first match wins).

  Keys of `rectification`, `split_key`, `activation`, `constrain_weight`, and `constrain_bias`
  are glob patterns (see utils.collections.GlobDict) matched against ObsKeys; observation keys
  matching no `rectification` pattern (or matching one mapped to None) pass through unmodified.
  `constrain_weight`/`constrain_bias` values are constrain callables applied to the corresponding
  parameter; unmatched keys are unconstrained.

  Rectifications (`RECTIFY_SUFFIX`):
  - 'sign': `[ReLU(-x), ReLU(x)]`, two nonnegative channels.
  - 'flip': `[high - x + low, x]`, the value mirrored within its box and the value itself, so the
    two channels rise in opposite directions. Both are nonnegative only if `low >= 0`.
  - 'activation': the value itself, through `activation` only.
  The channels are concatenated under the original key (doubling its size), or with `split_key`,
  stored as separate keys `{key}[<suffix>]` (e.g. `body/proprio_pos_lumbarBT[<-]` and `[->]`,
  which the provided configs split so the PF can treat the two directions of lumbar bend and twist
  as a mirrored pair of segments).
  """
  def __init__(
    self,
    rectification: T.Mapping[ObsKey, RectifyMode | None],
    split_key: T.Mapping[ObsKey, bool] | None = None,
    weight: bool = True,
    bias: bool = True,
    activation: T.Mapping[ObsKey, T.Callable[[], torch.nn.Module]] | None = None,
    constrain_weight: T.Mapping[ObsKey, ConstrainFn] | None = None,
    constrain_bias: T.Mapping[ObsKey, ConstrainFn] | None = None,
  ):
    """Initializes the rectifier.

    Args:
      rectification: Rectification per observation key pattern, or `None` to pass it through.
      split_key: Whether to store the channels of a key pattern as separate keys (see the class).
      weight: Whether each rectified key gets a learnable elementwise weight (initialized to 1).
      bias: Whether each rectified key gets a learnable elementwise bias (initialized to 0).
      activation: Activation module factory per rectified key pattern.
      constrain_weight: Constraint function per weight key pattern (e.g. `constrain`).
      constrain_bias: Constraint function per bias key pattern.
    """
    super().__init__()
    self.rectification = utc.GlobDict(rectification)
    self.split_key = utc.GlobDict(split_key or {})
    self.weight = torch.nn.ParameterDict() if weight else None
    self.bias = torch.nn.ParameterDict() if bias else None
    self.activation_fn = utc.GlobDict(activation or {})
    self.constrain_weight = utc.GlobDict(constrain_weight or {})
    self.constrain_bias = utc.GlobDict(constrain_bias or {})

  def initialize(self, observation_space: gymnasium.spaces.Dict) -> gymnasium.spaces.Dict:
    """Creates the parameters and activations, and returns the rectified observation space.

    Raises:
      ValueError: If a pattern of `constrain_weight`, `constrain_bias` or `activation` matches no
        observation key, or a labeled box is not 1-D.
    """
    self.observation_space = observation_space
    self.activation = torch.nn.ModuleDict()
    for label, patterns in (
      ('constrain_weight', self.constrain_weight),
      ('constrain_bias', self.constrain_bias),
      ('activation', self.activation_fn),
    ):
      unmatched = [
        p for p in patterns if not any(uts.in_glob_range(p, oname) for oname in observation_space)
      ]
      if unmatched:
        raise ValueError(f'{label} patterns match no observation keys: {unmatched}')
    observation_space_transformed = collections.OrderedDict(observation_space)
    for oname in observation_space:
      rectification = self.rectification.get(oname)
      if rectification is None:
        continue
      obox = observation_space[oname]
      assert isinstance(obox, gymnasium.spaces.Box)
      assert obox.shape is not None
      # Transform observation space based on the rectification type.
      low, high = obox.low, obox.high
      match rectification:
        case 'sign':
          low_neg = np.minimum(np.maximum(-low, 0), np.maximum(-high, 0))
          high_neg = np.maximum(np.maximum(-low, 0), np.maximum(-high, 0))
          low_pos = np.minimum(np.maximum(low, 0), np.maximum(high, 0))
          high_pos = np.maximum(np.maximum(low, 0), np.maximum(high, 0))
          lows, highs = [low_neg, low_pos], [high_neg, high_pos]
        case 'flip':
          lows, highs = [low, low], [high, high]
        case 'activation':
          lows, highs = [low], [high]
      suffixes = RECTIFY_SUFFIX[rectification]
      if isinstance(obox, cxg.Box) and obox.labels:
        if len(obox.labels) != 1:
          raise ValueError(f'Expected a 1-long label for 1-D box {obox!r}, got {obox.labels!r}.')
        labels = [obox.labels[0]] * len(suffixes)
      else:
        labels = [None] * len(suffixes)
      if not self.split_key.get(oname, False):
        lows, highs = [np.concatenate(lows)], [np.concatenate(highs)]
        suffixes, labels = [''], [None]
      for low, high, suffix, label in zip(lows, highs, suffixes, labels):
        oname_suffix = f'{oname}[{suffix}]' if suffix else oname
        if label and suffix:
          label = (
            f'{label}[{suffix}]' if isinstance(label, str) else [f'{l}[{suffix}]' for l in label]
          )
        observation_space_transformed[oname_suffix] = cxg.Box(
          low=low,
          high=high,
          shape=low.shape,
          dtype=obox.dtype,  # type: ignore
          labels=(label,) if label else None,
        )
        # Initialize learnable parameters for the rectification if applicable.
        if self.weight is not None:
          self.weight[oname_suffix] = torch.ones(low.shape)
          constrain_weight = self.constrain_weight.get(oname_suffix)
          if constrain_weight is not None:
            constrain_weight(self.weight[oname_suffix])
        if self.bias is not None:
          self.bias[oname_suffix] = torch.zeros(low.shape)
          constrain_bias = self.constrain_bias.get(oname_suffix)
          if constrain_bias is not None:
            constrain_bias(self.bias[oname_suffix])
        act = self.activation_fn.get(oname_suffix)
        if act is not None:
          self.activation[oname_suffix] = act()
    return gymnasium.spaces.Dict(observation_space_transformed)

  def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Returns the rectified observations of one sample (unmatched keys unchanged)."""
    observation = observation.copy()
    activation = self.activation
    for oname in self.observation_space:
      rectification = self.rectification.get(oname)
      if rectification is None:
        continue
      # Apply rectification.
      value = observation[oname]
      match rectification:
        case 'sign':
          values = [torch.clamp(-value, min=0), torch.clamp(value, min=0)]
        case 'flip':
          box = T.cast(gymnasium.spaces.Box, self.observation_space[oname])
          low, high = (
            torch.as_tensor(box.low, dtype=value.dtype),
            torch.as_tensor(box.high, dtype=value.dtype),
          )
          values = [high - value + low, value]
        case 'activation':
          values = [value]
        case None:
          pass
      suffixes = RECTIFY_SUFFIX[rectification]
      # Apply learnable affine transformation if applicable.
      if not self.split_key.get(oname, False):
        values, suffixes = [torch.cat(values, dim=-1)], ['']
      for value, suffix in zip(values, suffixes):
        oname_suffix = f'{oname}[{suffix}]' if suffix else oname
        if self.weight is not None:
          value = value * self.weight[oname_suffix]
        if self.bias is not None:
          value = value + self.bias[oname_suffix]
        if oname_suffix in activation:
          value = activation[oname_suffix](value)
        observation[oname_suffix] = value
    return observation


# ==================================================================================================
# RhythmGenerationCircuitObserver

LimbName = T.Literal['FL', 'FR', 'HL', 'HR']
"""A limb, as RG unit names suffix it: front/hind, left/right."""


class RhythmGenerationCircuitObserver():
  """
  Merges RhythmGeneration outputs into observations and correspondingly expands the observation
  space with labeled boxes, e.g. `{'rg_limbFL': Box(labels=['flx', 'flx2', 'ext', 'ext2'], ...)}`.

  The RG output units are named `{polarity}{index}_{limb}`, e.g. `flx_FL` and `flx2_FL`, where the
  optional index tells a polarity's outputs apart (the code calls it `delay`; see
  `RhythmGenerationCircuit` for what `flx2` and `ext2` compute).
  """
  def initialize(
    self,
    rhythm_generation: 'RhythmGenerationCircuit',
    observation_space: gymnasium.spaces.Dict,
    observation_prefix: str = 'rg',
  ) -> gymnasium.spaces.Dict:
    """Records the RG's output structure, and returns the observation space with one
    `{observation_prefix}_<limb>` box per limb added, labeled in the RG's output order.

    Raises:
      ValueError: If an output unit is not a limb's `flx`/`ext` unit, is duplicated, or the left
        and right limbs of a pair have different outputs.
    """
    self.observation_prefix = observation_prefix
    # Parse RG output structure information for clean downstream reorganization.
    # NOTE: The RG output units are assumed to have a specific naming convention:
    #       f'{polarity}{index}_{limb}' (the index is called `delay` below).
    output_units = rhythm_generation.get_output_units()  # flx_FL, flx2_FL, ext_FL, ext2_FL, ...
    output_structure: dict[str, dict[tuple[str, int], int]] = {
      k: {} for k in ['limbFL', 'limbFR', 'limbHL', 'limbHR']
    }  # LimbName: {(polarity, delay): index}
    limb_units = dict(FL='limbFL', FR='limbFR', HL='limbHL', HR='limbHR')
    for unit in output_units:
      role, limb_rg = unit.split('_')
      limb = limb_units[limb_rg]
      polarity = role[:3]
      delay = int(role[3:]) if len(role) > 3 else 0
      if not (polarity in ('flx', 'ext') and limb_rg in ('FL', 'FR', 'HL', 'HR')):
        raise ValueError(f'Unexpected RG output unit: {unit}.')
      if (polarity, delay) in output_structure[limb]:
        raise ValueError(f'Duplicate RG output unit for limb {limb} and delay {delay}.')
      # NOTE: `rhythm_generation.get_output()` returns per-limb tensors (local indexing), not a
      #       single global tensor, so store each unit's position within its own limb here rather
      #       than its position in the flattened `output_units` list.
      output_structure[limb][(polarity, delay)] = len(output_structure[limb])
    # Verify that contralateral limbs have the same RG units, since AF & PF share those weights.
    units_FL, units_FR = output_structure['limbFL'].keys(), output_structure['limbFR'].keys()
    if units_FL != units_FR:
      raise ValueError(f'RG units for FL and FR must match, but {units_FL} != {units_FR}')
    units_HL, units_HR = output_structure['limbHL'].keys(), output_structure['limbHR'].keys()
    if units_HL != units_HR:
      raise ValueError(f'RG units for HL and HR must match, but {units_HL} != {units_HR}')
    # Format the labels, in the RG's output order.
    self.output_mapping: dict[str, list[tuple[str, int]]] = {
      k: [
        (f'{polarity}{delay or ""}', index) for (polarity, delay), index in struct.items()
      ] for k, struct in output_structure.items()
    }  # LimbName: [(label, index), ...]
    observation_space = gymnasium.spaces.Dict(list(observation_space.items()))
    for limb, mapping in self.output_mapping.items():
      observation_space[f'{self.observation_prefix}_{limb}'] = cxg.Box(
        low=0.0,
        high=1.0,
        shape=(len(mapping),),
        labels=[[l for l, _ in mapping]]
      )
    return observation_space

  def insert(
    self,
    rhythm_generation_output: T.Mapping[str, torch.Tensor],
    observation: T.Mapping[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    """Returns the observations with the RG outputs added as `{observation_prefix}_<limb>`."""
    observation = dict(observation)
    for limb, mapping in self.output_mapping.items():
      limb_rg = limb[-2:]
      indices = torch.tensor([index for _, index in mapping])
      if limb_rg in rhythm_generation_output:
        out = rhythm_generation_output[limb_rg][indices]
      else:
        assert len(mapping) == 0, (
          f'Expected no RG output for limb {limb}, but got indices {indices}.'
        )
        out = torch.empty((0,))
      observation[f'{self.observation_prefix}_{limb}'] = out
    return observation


# ==================================================================================================
# Rhythm Generation module and initializers


class RhythmGenerationOscKwargs(T.TypedDict, total=False):
  """Keyword arguments of the RG's `cc.Oscillator` units (see `circuits.Oscillator`)."""
  activation: T.Callable
  adaptation_time: float
  active_time: float
  quiet_time: float
  active_scale_pos: float
  quiet_scale_pos: float
  active_scale_neg: float
  quiet_scale_neg: float
  adaptation_scale_active_pos: float
  adaptation_scale_quiet_pos: float
  adaptation_scale_active_neg: float
  adaptation_scale_quiet_neg: float
  active_stable: float
  quiet_stable: float
  active_stable_margin: float
  quiet_stable_margin: float
  active_delay: float
  quiet_delay: float
  noise: float


class RhythmGenerationBasicKwargs(T.TypedDict, total=False):
  """Keyword arguments of the RG's `cc.Basic` units (see `circuits.Basic`)."""
  activation: T.Callable
  voltage_time: float


class RhythmGenerationCircuitInitializer(T.Protocol):
  """Initializer of a `RhythmGenerationCircuit` (e.g. `init_rg_rat`)."""
  def __call__(
    self,
    p: utt.containers.ParameterManager,
    grad: bool = False,
  ) -> tuple[RhythmGenerationOscKwargs, RhythmGenerationOscKwargs, RhythmGenerationBasicKwargs]:
    """Configures the RG's parameters (unit biases, connection weights) in `p`, and returns the
    unit hyperparameters: `(flexor kwargs, extensor kwargs, interneuron kwargs)`. The extensor
    kwargs must match the circuit's `ext_oscillator`: Oscillator kwargs if true, Basic kwargs if
    false."""
    ...


def logarithmic_mod_amplitude(
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
  """`circuits.logarithmic` with its active amplitude scaled from 0.6 to 1 as the input `x` goes
  from 0.5 to 0.7 (an `Oscillator` activation; see `circuits.units`)."""
  pos = cc.clip_normalized(0.5, 0.7, x)
  mag = cc.interpolate_linear(0.6, 1, pos)
  return (
    (v >= 0) * mag * cc.interpolate_logarithmic(
      active_end, active_start, cc.clip_normalized(active_bound, quiet_bound, a), base=base
    ) + (v < 0) * quiet
  )


def activation_locomotion(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  mod_amplitude: bool = False,
) -> torch.Tensor:
  """`circuits.logarithmic` (or `logarithmic_mod_amplitude`) with the active output starting at
  0.85-0.90 and ending at 0.05-0.2, and the quiet output 0-0.1, each rising with the input `x`
  (an `Oscillator` activation of `init_rg_mouse`)."""
  pos = torch.clip(x, 0, 1)
  return (logarithmic_mod_amplitude if mod_amplitude else cc.logarithmic)(
    v,
    a,
    x,
    active_bound=active_bound,
    quiet_bound=quiet_bound,
    active_start=cc.interpolate_linear(0.85, 0.90, cc.clip_normalized(0.0, 1.0, pos)),
    active_end=cc.interpolate_linear(0.05, 0.2, cc.clip_normalized(0.0, 1.0, pos)),
    quiet=cc.interpolate_linear(0, 0.1, cc.clip_normalized(0.6, 1.0, pos)),
    base=20,
  )


def activation_locomotion4_flx(
  v: torch.Tensor,
  a: torch.Tensor,
  x: torch.Tensor,
  active_bound: torch.Tensor,
  quiet_bound: torch.Tensor,
  *,
  active_start: torch.Tensor | float = 1.0,
  active_end: torch.Tensor | float = 0.5,
  amplitude_start: torch.Tensor | float = 0.5,
  amplitude_end: torch.Tensor | float = 1.0,
  quiet: torch.Tensor | float = 0.,
  base: float = 2.,
) -> torch.Tensor:
  """The flexor `Oscillator` activation of `init_rg_rat`: when active, decays logarithmically
  (with `base`) from `active_start` to `active_end` over the active phase, scaled by an amplitude
  that rises linearly from `amplitude_start` to `amplitude_end` as the input `x` goes from 0 to 1;
  `quiet` when quiet. The active phase is normalized by the fixed bounds that the Oscillator
  passes. The output can exceed 1 (e.g. 1.3 with `active_start=1.3`)."""
  pos = cc.clip_normalized(0., 1., x)
  amp = cc.interpolate_linear(amplitude_start, amplitude_end, pos)
  return (
    (v >= 0) * amp * cc.interpolate_logarithmic(
      active_end, active_start, cc.clip_normalized(active_bound, quiet_bound, a), base=base
    ) + (v < 0) * quiet
  )


def activation_locomotion4_ext(
  v: torch.Tensor,
  *,
  amplitude_start: torch.Tensor | float = 0.5,
  amplitude_end: torch.Tensor | float = 1.0,
) -> torch.Tensor:
  """The extensor `Basic` activation of `init_rg_rat`, `activation(v)`: for a nonnegative voltage,
  an amplitude rising linearly from `amplitude_start` to `amplitude_end` as `v` goes from 0 to 1;
  0 otherwise."""
  pos = cc.clip_normalized(0., 1., v)
  amp = cc.interpolate_linear(amplitude_start, amplitude_end, pos)
  return (v >= 0) * amp


def init_rg_default(p: utt.containers.ParameterManager, grad: bool = False):
  """RG weight initialization (default): Used for Fixed Speed experiments."""
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
  osc: RhythmGenerationOscKwargs = {
    'adaptation_time': 1400,
    'active_time': 200,
    'quiet_time': 1200,
    'active_scale_pos': 100 / 200,
    'quiet_scale_pos': 60 / 1200,
    'active_stable': 3,
  }
  basic: RhythmGenerationBasicKwargs = {
    'voltage_time': 30,
  }
  return osc, osc, basic  # flx = ext


def init_rg_full(p: utt.containers.ParameterManager, grad: bool = False):
  """RG weight initialization (full): Used for Variable Speed experiments to enable the full range
  of BC commands to work a single RG module.

  Major changes compared to `init_rg_default`:
  1. Updated `bias_flx`/`bias_ext` to ensure defaults to quiet mode without input.
  2. Added `freq_ext_e` to allow BC to control extensor activity.
  3. Updated Oscillator unit `activation` function so amplitude increase with increasing input.
  """
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], -0.2, grad)  # bc excite
  p.config('bias_ext', ['bias_ext_*'], -0.2, grad)  # bc excite
  p.config('bias_v0d', ['bias_v0d_*'], -0., grad)  # active at slow (cross async)
  p.config('bias_v0v', ['bias_v0v_*'], -0., grad)  # active at slow (diag async)
  p.config('bias_v3e', ['bias_v3e_*'], -0.5, grad)  # inactive at slow (cross sync)
  p.config('bias_v3f', ['bias_v3f_*'], -0.5, grad)  # inactive at slow (cross sync)
  p.config('bias_v3a', ['bias_v3a_*'], -0.2, grad)  # inactive at slow (diag sync)
  p.config('bias_in2', ['bias_in2_*'], -0.2, grad)  # inactive at slow (diag sync)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -1.5, grad)  # flx dominates, osc async
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.5, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx_i_R', ['cross_flxflx_i_*R'], -1.4, grad)  # cross async at slow, bc inhibit
  p.config('cross_flxflx_i_L', ['cross_flxflx_i_*L'], -1.2, grad)  # promotes async, bc inhibit
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.2, grad)  # cross sync at fast, bc excite
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.1, grad)  # cross sync
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -0.4, grad)  # side async
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.1, grad)  # side async
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.1, grad)  # side async
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -0.5, grad)  # diag async at slow
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.5, grad)  # diag async at slow
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.1, grad)  # diag sync at fast
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.1, grad)  # diag sync at fast
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1., False)  # identity
  p.config('aff_ext', ['aff_ext_*'], 1., False)  # identity
  # Command connections (freq, sync).
  p.config('freq_osc_e', ['freq_osc_e_*'], 1., grad)  # identity (flexor/oscillator)
  p.config('freq_ext_e', ['freq_ext_e_*'], 0.8, grad)  # identity (extensor)
  p.config('freq_cross_e', ['freq_cross_e_*'], 0.2, grad)  # cross sync at fast
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.5, grad)  # hind-driven at fast
  p.config('sync_cross_i', ['sync_cross_i_*'], -1.2, grad)  # cross sync at fast
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -0.5, grad)  # diag async at slow
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.3, grad)  # diag sync at fast
  # Unit parameters.
  osc: RhythmGenerationOscKwargs = {
    'activation': cc.rectangular_amplitude_modulated,
    'adaptation_time': 1600,
    'active_time': 200,
    'quiet_time': 1400,
    'active_scale_pos': 100 / 200,
    'quiet_scale_pos': 80 / 1400,
    'active_stable': 3,
  }
  basic: RhythmGenerationBasicKwargs = {
    'voltage_time': 30,
  }
  return osc, osc, basic  # flx = ext


def init_rg_mouse(
  p: utt.containers.ParameterManager,
  grad: bool = False,
  mod_amplitude: bool = False,
):
  """RG weight initialization (full): Used for OSC circuit experiments based on Mouse data.

  Major changes compared to `init_rg_full`:
  1. Swap flx to Oscillator.
  2. Removed `bias_ext`, `freq_ext_e`, and `amplitude` modulation to reproduce Danner et al. 2017.
  3. Retuned weights to produce appropriate gait transitions.
  """
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], -0.05, grad)  # flx quiet at baseline bc
  p.config('bias_ext', ['bias_ext_*'], 1.2, grad)  # ext tonic at baseline bc
  p.config('bias_v0d', ['bias_v0d_*'], -0., grad)  # active at slow (cross async)
  p.config('bias_v3e', ['bias_v3e_*'], -0., grad)  # inactive at slow (cross sync)
  p.config('bias_v3f', ['bias_v3f_*'], -0.2, grad)  # inactive at slow (cross sync)
  p.config('bias_v0v', ['bias_v0v_*'], -0., grad)  # active at slow (diag async)
  p.config('bias_v3a', ['bias_v3a_*'], -0.7, grad)  # inactive at slow (diag sync)
  p.config('bias_in2', ['bias_in2_*'], -0.5, grad)  # inactive at slow (diag sync)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -2.0, grad)  # flx dominates, osc async
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.1, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx_i_R', ['cross_flxflx_i_*R'], -1.2, grad)  # cross async at slow, bc inhibit
  p.config('cross_flxflx_i_L', ['cross_flxflx_i_*L'], -1.4, grad)  # promotes async, bc inhibit
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.2, grad)  # cross sync at fast, bc excite
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.2, grad)  # cross sync
  p.config('cross_extflx_i', ['cross_extflx_i_*'], -0.3, grad)  # cross walk, left-right phase 0.5
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -1.2, grad)  # side async
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.05, grad)  # side async
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.05, grad)  # side async
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -0.8, grad)  # diag async at slow
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.7, grad)  # diag async at slow
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.1, grad)  # diag sync at fast
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1., False)  # identity
  p.config('aff_ext', ['aff_ext_*'], 1., False)  # identity
  # Command connections (freq, sync).
  p.config('freq_flx_e_F', ['freq_flx_e_F*'], 0.65, grad)  # flx rises
  p.config('freq_flx_e_H', ['freq_flx_e_H*'], 0.65, grad)  # flx rises
  p.config('freq_ext_e', ['freq_ext_e_*'], 0., grad)  # ext constant
  p.config('freq_cross_e', ['freq_cross_e_*'], 0.3, grad)  # cross sync at fast (v3f)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.5, grad)  # diag sync at trot (v3a)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.6, grad)  # cross sync at bound (v0d)
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -1.2, grad)  # diag async walk, sync trot (v0d)
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.6, grad)  # diag sync trot, async bound (v0v)

  # Unit hyperparameters.
  osc: RhythmGenerationOscKwargs = {
    'activation': functools.partial(activation_locomotion, mod_amplitude=mod_amplitude),
    'adaptation_time': 450,
    'active_time': 75,
    'quiet_time': 400,
    'active_scale_pos': 0.125,
    'quiet_scale_pos': 0.025,
    'active_scale_neg': 0.4,
    'quiet_scale_neg': 1.1,
    'adaptation_scale_active_pos': 1.,
    'adaptation_scale_quiet_pos': 1.2,
    'adaptation_scale_active_neg': 1.2,
    'adaptation_scale_quiet_neg': 0.8,
    'active_stable': 1,
    'quiet_delay': 0.1,
    'quiet_stable_margin': 50,
    'noise': 0.02,
  }
  basic: RhythmGenerationBasicKwargs = {
    'voltage_time': 10,
  }
  return osc, osc, basic  # flx = ext


def init_rg_rat(
  p: utt.containers.ParameterManager,
  grad: bool = False,
  grad_flx: bool = False,
  grad_ext: bool = False,
  scale: float = 0.8,
  active: float = 200,
  quiet: float = 600,
  adaptation_scale: float = 1.5,
):
  """The rat RG initialization (`RhythmGenerationCircuitInitializer`), with phase durations from
  rat locomotion data, used by the rat NCAP models.

  Parameter names follow `{pathway}_{source}{target}_{e|i}` (excitatory or inhibitory), configured
  for unit names matching a glob pattern, so units matching one pattern share one parameter:
  - pathways: `osc` (flexor-extensor half-center of one limb), `cross` (left-right), `sided`/
    `sidea` (same side, descending/ascending), `diagd`/`diaga` (diagonal, descending/ascending),
    `aff` (afferent input), `freq`/`sync` (BC command input); `bias_<unit type>` are unit biases;
  - e.g. `cross_flxflx_i_*R` are the inhibitions of the flexors onto the right-side flexors.
  The left-right inhibition between flexors is deliberately asymmetric (`cross_flxflx_i_R` -1.7,
  `cross_flxflx_i_L` -2.5), which pairs with the circuit's per-limb `flx_delays`. The afferent
  weights `aff_*` are fixed at 1 (identity), whatever `grad`.

  Args:
    p: Parameter manager to configure.
    grad: Whether the interneuron biases and the connection weights are trained.
    grad_flx: Whether the flexor biases are trained.
    grad_ext: Whether the extensor biases are trained.
    scale: Factor of the active, quiet and adaptation times.
    active: Active (flexion) phase duration before `scale` (ms).
    quiet: Quiet phase duration before `scale` (ms).
    adaptation_scale: Factor of the adaptation time relative to the longer phase.

  Returns:
    `(flexor kwargs, extensor kwargs, interneuron kwargs)`: flexors are Oscillators with
    `activation_locomotion4_flx`; extensors are Basic units with `activation_locomotion4_ext`, so
    the circuit needs `ext_oscillator=False`. The flexors' `quiet_delay` here is replaced by the
    circuit's `flx_delays`.
  """
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], -0.05, grad_flx)  # flx quiet at baseline bc
  p.config('bias_ext', ['bias_ext_*'], -0.05, grad_ext)  # ext quiet at baseline bc
  p.config('bias_v0d', ['bias_v0d_*'], -0., grad)  # active at slow (cross async)
  p.config('bias_v3e', ['bias_v3e_*'], -0., grad)  # inactive at slow (cross sync)
  p.config('bias_v3f', ['bias_v3f_*'], -0.2, grad)  # inactive at slow (cross sync)
  p.config('bias_v0v', ['bias_v0v_*'], -0., grad)  # active at slow (diag async)
  p.config('bias_v3a', ['bias_v3a_*'], -0.5, grad)  # inactive at slow (diag sync)
  p.config('bias_in2', ['bias_in2_*'], -0.5, grad)  # inactive at slow (diag sync)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -2.0, grad)  # flx dominates, osc async
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.1, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx_i_R', ['cross_flxflx_i_*R'], -1.7, grad)  # cross async at slow, bc inhibit
  p.config('cross_flxflx_i_L', ['cross_flxflx_i_*L'], -2.5, grad)  # cross async at slow, bc inhibit
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.2, grad)  # cross sync at fast, bc excite
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.2, grad)  # cross sync at fast, bc excite
  p.config('cross_extflx_i', ['cross_extflx_i_*'], -0.01, grad)  # cross walk, left-right phase 0.5
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -1.2, grad)  # side async
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.05, grad)  # side async
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.05, grad)  # side async
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -1.2, grad)  # diag async at slow
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.7, grad)  # diag async at slow
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.4, grad)  # diag sync at fast
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.1, grad)  # diag sync at fast
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1., False)  # identity
  p.config('aff_ext', ['aff_ext_*'], 1., False)  # identity
  # Command connections (freq, sync).
  p.config('freq_flx_e_F', ['freq_flx_e_F*'], 0.82, grad)  # flx rises
  p.config('freq_flx_e_H', ['freq_flx_e_H*'], 0.85, grad)  # flx rises
  p.config('freq_ext_e', ['freq_ext_e_*'], 0.9, grad)  # ext constant
  p.config('freq_cross_e', ['freq_cross_e_*'], 0.2, grad)  # cross sync at bound (v3f)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.3, grad)  # diag sync at trot (v3a)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.8, grad)  # cross sync at bound (v0d)
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -0.9, grad)  # diag async walk -> sync trot (v0d)
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.6, grad)  # diag sync trot -> async bound (v0v)

  # Unit hyperparameters.
  osc: RhythmGenerationOscKwargs = {
    # New hyperparameters based on rat data.
    'adaptation_time': max(active, quiet) * scale * adaptation_scale,
    'active_time': active * scale,
    'quiet_time': quiet * scale,
    'active_scale_pos': 0.5,
    'quiet_scale_pos': 0.1,
    # Same hyperparameters as `init_rg_mouse`.
    'active_scale_neg': 0.4,
    'quiet_scale_neg': 1.1,
    'adaptation_scale_active_pos': 1.,
    'adaptation_scale_quiet_pos': 1.2,
    'adaptation_scale_active_neg': 1.2,
    'adaptation_scale_quiet_neg': 0.8,
    'quiet_delay': 0.1,
    'quiet_stable_margin': 50,
    'noise': 0.02,
  }
  flx: RhythmGenerationOscKwargs = {
    **osc,
    'activation': functools.partial(
      activation_locomotion4_flx,
      active_start=1.3,
      active_end=0.4,
      amplitude_start=0.2,
      amplitude_end=1.0,
    ),
    'active_stable': 3,  # Never allow flx to remain in tonic (locked flexors).
  }
  ext: RhythmGenerationBasicKwargs = {
    'activation': functools.partial(
      activation_locomotion4_ext, amplitude_start=0.2, amplitude_end=1.0,
    ),
    'voltage_time': 30,
  }
  basic: RhythmGenerationBasicKwargs = {
    'voltage_time': 10,
  }

  return flx, ext, basic


def _conditional_set(
  field: torch.Tensor,
  value: torch.Tensor,
  condition: torch.Tensor | None = None,
):
  """Sets `field` to `value` where `condition` is true, otherwise leaves it unchanged."""
  field[:] = value if condition is None else (condition * value + ~condition * field)


class RhythmGenerationCircuit(cc.Group):
  """The Rhythm Generation (RG) circuit module coordinates gait between limbs. It is implemented
  using the `circuits` library by connecting Oscillator and Basic units according to genetically
  defined neural circuit models (Danner et al. 2019). The limbs are denoted "FL" (front left), "FR"
  (front right), "HL" (hind left), and "HR" (hind right).

  Units (attributes), per limb `<limb>` unless noted:
  - `flx_<limb>`: flexor half-center, an `Oscillator`.
  - `ext_<limb>`: extensor half-center, a `Basic` unit if `ext_oscillator` is false (as the rat
    models), else an `Oscillator`.
  - `cross_v0d_<limb>`, `cross_v3f_<limb>`, `cross_v3e_<limb>`: commissural interneurons, which
    couple the left and right sides.
  - `diagd_v0d_<limb>`, `diagd_v0v_<limb>`, `diagd_in2_<limb>` (forelimbs only): descending
    diagonal interneurons; `diaga_v3a_<limb>` (hindlimbs only): ascending diagonal interneurons.
  - `aff_flx_<limb>`, `aff_ext_<limb>`: `Signal`s of the afferent inputs (written by the actor
    from the AF's reset outputs).
  - `freq_{L,R}`, `sync_{L,R}`: `Signal`s of the brainstem command per side (written by the actor
    from the BC's `speed`).
  - `flx2_<limb>`: a `Basic` copy of `flx_<limb>` through a 2-step synapse, so a delayed and
    low-passed (not exact) copy.
  - `ext2_<limb>`: a `Basic` unit of `ext_<limb> - flx_<limb>` (bias -0.3, clamped to [0, 1]),
    active late in the extension phase.
  The outputs are, per limb, `[flx, flx2, ext, ext2]` (`get_output_units`).

  Every synapse has the default 1-step delay unless noted. Synapses' `sign=` is rendering
  metadata: a connection's sign comes from its initialized weight, and is kept by constraints
  (if any), not by the circuit. Parameters that the initializer does not configure are the frozen
  `zero` parameter. Unit and parameter names are state_dict keys, and the recorders and ablations
  refer to them, so they are part of the interface. The layout constants and `xy`, `flow`,
  `anchor` and `waypoints` arguments only place the units in circuit diagrams.
  """
  def __init__(
    self,
    init: RhythmGenerationCircuitInitializer = init_rg_mouse,
    ext_oscillator: bool = True,
    flx_delays: T.Sequence[float] = (0.07, 0.02, 0.07, 0.02),
    **kwargs,
  ):
    """Initializes the circuit. The defaults are the original (mouse) circuit's; the rat models
    pass `init_rg_rat`, `ext_oscillator=False` and their own `flx_delays`.

    Args:
      init: Initializer of the parameters and unit hyperparameters.
      ext_oscillator: Whether the extensors are Oscillators (else Basic units). Must match the
        extensor kwargs that `init` returns.
      flx_delays: Each flexor's `quiet_delay` (FL, FR, HL, HR): the fraction of its quiet phase
        spent as a delay before it can turn active, in [0, 1). Not a time delay.
      **kwargs: Metadata of the group node (e.g. for rendering). Unknown arguments end up here
        without an error.
    """
    super().__init__(**kwargs)
    # ----------------------------------------------------------------------------------------------
    # Parameters

    self.params = p = utt.containers.ParameterManager()
    p.config('zero', ['*'], 0., False)  # Parameters default to 0, unless initialized otherwise.
    flx_kwargs, ext_kwargs, basic_kwargs = init(p)

    # ----------------------------------------------------------------------------------------------
    # Circuit units

    # Oscillator units.
    XF = -20
    XH = -70
    XFLX = 0
    YFLX = 21
    XEXT = 0
    YEXT = 24
    # FL
    self.flx_FL = cc.Oscillator(
      bias=p['bias_flx_FL'],
      **{
        **flx_kwargs, 'quiet_delay': flx_delays[0]
      },
      xy=(XF - XFLX, +YFLX),
      anchor='y+',
    )
    if ext_oscillator:
      self.ext_FL = cc.Oscillator(bias=p['bias_ext_FL'], **ext_kwargs, xy=(XF - XEXT, +YEXT))
    else:
      self.ext_FL = cc.Basic(bias=p['bias_ext_FL'], **ext_kwargs, xy=(XF - XEXT, +YEXT))
    # FR
    self.flx_FR = cc.Oscillator(
      bias=p['bias_flx_FR'],
      **{
        **flx_kwargs, 'quiet_delay': flx_delays[1]
      },
      xy=(XF - XFLX, -YFLX),
      anchor='y+'
    )
    if ext_oscillator:
      self.ext_FR = cc.Oscillator(bias=p['bias_ext_FR'], **ext_kwargs, xy=(XF - XEXT, -YEXT))
    else:
      self.ext_FR = cc.Basic(bias=p['bias_ext_FR'], **ext_kwargs, xy=(XF - XEXT, -YEXT))
    # HL
    self.flx_HL = cc.Oscillator(
      bias=p['bias_flx_HL'],
      **{
        **flx_kwargs, 'quiet_delay': flx_delays[2]
      },
      xy=(XH - XFLX, +YFLX),
      anchor='y+'
    )
    if ext_oscillator:
      self.ext_HL = cc.Oscillator(bias=p['bias_ext_HL'], **ext_kwargs, xy=(XH - XEXT, +YEXT))
    else:
      self.ext_HL = cc.Basic(bias=p['bias_ext_HL'], **ext_kwargs, xy=(XH - XEXT, +YEXT))
    # HR
    self.flx_HR = cc.Oscillator(
      bias=p['bias_flx_HR'],
      **{
        **flx_kwargs, 'quiet_delay': flx_delays[3]
      },
      xy=(XH - XFLX, -YFLX),
      anchor='y+'
    )
    if ext_oscillator:
      self.ext_HR = cc.Oscillator(bias=p['bias_ext_HR'], **ext_kwargs, xy=(XH - XEXT, -YEXT))
    else:
      self.ext_HR = cc.Basic(bias=p['bias_ext_HR'], **ext_kwargs, xy=(XH - XEXT, -YEXT))

    # Derived output units (flx2, ext2): hidden in circuit diagrams (`visible=False`: the unit and
    # its connections are not drawn; it has no diagram position).
    self.delay_units = []
    for leg in ['FL', 'FR', 'HL', 'HR']:
      name = f'flx2_{leg}'
      setattr(self, name, cc.Basic(visible=False).synapse(
        getattr(self, f'flx_{leg}'),
        weight=1.0,
        delay=2,
      ))
      self.delay_units.append(name)
      name = f'ext2_{leg}'
      setattr(self, name, cc.Basic(
        bias=-0.3,
        activation=lambda x: torch.clamp(x, min=0, max=1),
        voltage_time=80,
        visible=False,
      ).synapse(
        getattr(self, f'ext_{leg}'),
        weight=1.0,
        # delay=2,
      ).synapse(getattr(self, f'flx_{leg}'), weight=-1.0))
      self.delay_units.append(name)

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
    # FL
    self.cross_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XC_V0D, +YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_FL = cc.Basic(
      bias=p['bias_v3f_FL'], **basic_kwargs, xy=(XF - XC_V3F, +YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_FL = cc.Basic(
      bias=p['bias_v3e_FL'], **basic_kwargs, xy=(XF - XC_V3E, +YC_V3E), flow='x-'
    )
    self.diagd_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XD_V0D, +YD_V0D), anchor='y+'
    )
    self.diagd_v0v_FL = cc.Basic(
      bias=p['bias_v0v_FL'], **basic_kwargs, xy=(XF - XD_V0V, +YD_V0V), anchor='y+'
    )
    self.diagd_in2_FL = cc.Basic(
      bias=p['bias_in2_FL'], **basic_kwargs, xy=(XF - XD_IN2, +YD_IN2), anchor='y+'
    )
    # FR
    self.cross_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XC_V0D, -YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_FR = cc.Basic(
      bias=p['bias_v3f_FR'], **basic_kwargs, xy=(XF - XC_V3F, -YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_FR = cc.Basic(
      bias=p['bias_v3e_FR'], **basic_kwargs, xy=(XF - XC_V3E, -YC_V3E), flow='x-'
    )
    self.diagd_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XD_V0D, -YD_V0D), anchor='y+'
    )
    self.diagd_v0v_FR = cc.Basic(
      bias=p['bias_v0v_FR'], **basic_kwargs, xy=(XF - XD_V0V, -YD_V0V), anchor='y+'
    )
    self.diagd_in2_FR = cc.Basic(
      bias=p['bias_in2_FR'], **basic_kwargs, xy=(XF - XD_IN2, -YD_IN2), anchor='y+'
    )
    # HL
    self.cross_v0d_HL = cc.Basic(
      bias=p['bias_v0d_HL'], **basic_kwargs, xy=(XH - XC_V0D, +YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_HL = cc.Basic(
      bias=p['bias_v3f_HL'], **basic_kwargs, xy=(XH - XC_V3F, +YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_HL = cc.Basic(
      bias=p['bias_v3e_HL'], **basic_kwargs, xy=(XH - XC_V3E, +YC_V3E), flow='x-'
    )
    self.diaga_v3a_HL = cc.Basic(
      bias=p['bias_v3a_HL'], **basic_kwargs, xy=(XH - XA_V3A, +YA_V3A), flow='x-', anchor='y+'
    )
    # HR
    self.cross_v0d_HR = cc.Basic(
      bias=p['bias_v0d_HR'], **basic_kwargs, xy=(XH - XC_V0D, -YC_V0D), flow='x-', anchor='y+'
    )
    self.cross_v3f_HR = cc.Basic(
      bias=p['bias_v3f_HR'], **basic_kwargs, xy=(XH - XC_V3F, -YC_V3F), flow='x-', anchor='y+'
    )
    self.cross_v3e_HR = cc.Basic(
      bias=p['bias_v3e_HR'], **basic_kwargs, xy=(XH - XC_V3E, -YC_V3E), flow='x-'
    )
    self.diaga_v3a_HR = cc.Basic(
      bias=p['bias_v3a_HR'], **basic_kwargs, xy=(XH - XA_V3A, -YA_V3A), flow='x-', anchor='y+'
    )

    # Afferent units.
    XAF_FLX = XFLX - 5
    YAF_FLX = YFLX - 6
    YAF_FLXH = YFLX - 8
    XAF_EXT = XEXT - 5
    YAF_EXT = YEXT + 2
    # FL
    self.aff_flx_FL = cc.Signal(xy=(XF - XAF_FLX, +YAF_FLX), anchor='y+')
    self.aff_ext_FL = cc.Signal(xy=(XF - XAF_EXT, +YAF_EXT))
    # FR
    self.aff_flx_FR = cc.Signal(xy=(XF - XAF_FLX, -YAF_FLX), anchor='y+')
    self.aff_ext_FR = cc.Signal(xy=(XF - XAF_EXT, -YAF_EXT))
    # HL
    self.aff_flx_HL = cc.Signal(xy=(XH - XAF_FLX, +YAF_FLXH), anchor='y+')
    self.aff_ext_HL = cc.Signal(xy=(XH - XAF_EXT, +YAF_EXT))
    # HR
    self.aff_flx_HR = cc.Signal(xy=(XH - XAF_FLX, -YAF_FLXH), anchor='y+')
    self.aff_ext_HR = cc.Signal(xy=(XH - XAF_EXT, -YAF_EXT))

    # Command units.
    XCMD = -1
    YFREQ = 3
    YSYNC = 1
    # R
    self.freq_R = cc.Signal(xy=(XCMD, -YFREQ), anchor='y+')
    self.sync_R = cc.Signal(xy=(XCMD, -YSYNC))
    # L
    self.freq_L = cc.Signal(xy=(XCMD, +YFREQ), anchor='y+')
    self.sync_L = cc.Signal(xy=(XCMD, +YSYNC))

    # ----------------------------------------------------------------------------------------------
    # Circuit connections

    # Alias self to ensure that named access of circuit units typechecks correctly.
    u = T.cast(T.Mapping[str, cc.Basic | cc.Oscillator], self)

    # Oscillator connections (osc).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'ext_{a}'], weight=p[f'osc_extflx_i_{a}'], sign='-')
      u[f'ext_{a}'].synapse(u[f'flx_{a}'], weight=p[f'osc_flxext_i_{a}'], sign='-')

    # Cross-side connections (cross).
    for a, b in (('FR', 'FL'), ('FL', 'FR'), ('HR', 'HL'), ('HL', 'HR')):
      u[f'cross_v3f_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v3e_{a}'].synapse(u[f'ext_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(u[f'cross_v0d_{a}'], weight=p[f'cross_flxflx_i_{b}'], sign='-')
      u[f'flx_{b}'].synapse(
        u[f'cross_v3e_{a}'], weight=p[f'cross_extflx_i_{b}'], sign='-', waypoints=[(-8, +4, 'y-')]
      )
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
        u[f'diagd_v0d_{a}'],
        weight=p[f'diagd_flxflx_i_{b}'],
        sign='-',
        waypoints=[(-35, +12, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0v_{a}'],
        weight=p[f'diagd_flxflx_e_{b}'],
        sign='+',
        waypoints=[(-34, +12, 'y-')]
      )
      u[f'diagd_in2_{a}'].synapse(u[f'cross_v3f_{a}'], weight=1, sign='+')
      u[f'diagd_v0d_{a}'].synapse(u[f'diagd_in2_{a}'], weight=p[f'diagd_in2v0d_i_{a}'], sign='-')

    # Diagonal, ascending connections (diaga).
    for a, b in (('HL', 'FR'), ('HR', 'FL')):
      u[f'diaga_v3a_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diaga_v3a_{a}'],
        weight=p[f'diaga_flxflx_e_{b}'],
        sign='+',
        waypoints=[(32, 8.5, 'x-'), (-3, 7, 'y-')]
      )
      u[f'diagd_in2_{b}'].synapse(
        u[f'diaga_v3a_{a}'],
        weight=p[f'diaga_v3ain2_e_{b}'],
        sign='+',
        waypoints=[(26, -7.5, 'x-')]
      )

    # Afferent connections (aff).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'aff_flx_{a}'], weight=p[f'aff_flx_{a}'])
      u[f'ext_{a}'].synapse(u[f'aff_ext_{a}'], weight=p[f'aff_ext_{a}'])

    # Command connections (freq, sync).
    for a in ('L', 'R'):
      u[f'flx_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_flx_e_F{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'flx_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_flx_e_H{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'ext_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_F{a}'], sign='+', waypoints=[(-10, -15, 'y-')]
      )
      u[f'ext_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_ext_e_H{a}'], sign='+', waypoints=[(-9, -15, 'y-')]
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

  def reset(self, reset: torch.Tensor | None = None):
    """Sets each flexor's adaptation to a staggered fraction of its quiet bound (FL 0.7, FR 0.9,
    HL 0.8, HR 1.0), so the limbs start at different phases instead of settling from the same
    state. Call after the simulator's reset.

    Args:
      reset: Whether to reset, bool `[1]` (per sample when vmapped), or `None` to reset.
    """
    # Initialize flexor units with staggered adaptation variable to prevent settling period.
    _conditional_set(self.flx_FL.a, torch.tensor(0.7) * self.flx_FL.quiet_bound_zero, reset)
    _conditional_set(self.flx_FR.a, torch.tensor(0.9) * self.flx_FR.quiet_bound_zero, reset)
    _conditional_set(self.flx_HL.a, torch.tensor(0.8) * self.flx_HL.quiet_bound_zero, reset)
    _conditional_set(self.flx_HR.a, torch.tensor(1.0) * self.flx_HR.quiet_bound_zero, reset)

  def get_output_units(self) -> list[str]:
    """Returns the names of the output units, per limb `[flx, flx2, ext, ext2]`."""
    return sum([[
      f'flx_{limb}',
      *[d for d in self.delay_units if d.startswith('flx') and d.endswith(limb)],
      f'ext_{limb}',
      *[d for d in self.delay_units if d.startswith('ext') and d.endswith(limb)],
    ] for limb in ('FL', 'FR', 'HL', 'HR')], [])

  def get_output(self) -> 'dict[LimbName, torch.Tensor]':
    """Returns each limb's output values, in `get_output_units` order, {limb: [4]}."""
    return {
      limb:
        torch.concat(
          (
            T.cast(cc.Signal, self[f'flx_{limb}']).value,
            *[
              T.cast(cc.Signal, self[d]).value for d in self.delay_units
              if d.startswith('flx') and d.endswith(limb)
            ],
            T.cast(cc.Signal, self[f'ext_{limb}']).value,
            *[
              T.cast(cc.Signal, self[d]).value for d in self.delay_units
              if d.startswith('ext') and d.endswith(limb)
            ],
          ),
          dim=-1,
        )
      for limb in ('FL', 'FR', 'HL', 'HR')
    }


# ==================================================================================================
# Constrainer module and helpers

class Constrainer(torch.nn.Module):
  """A helper module that constrains `torch.nn.Parameter`/`torch.Tensor` values if they have defined
  constraint functions. After every parameter update, calling `constrain()` modifies values in-place
  to respect constraints. For example, this is useful to enforce sign constraints (excitatory vs.
  inhibitory) on synaptic weights.

  The constraints are the elementwise bounds that `constrain` (or `constrain_sign`,
  `constrain_bound`) attaches to a parameter as its `constrain_min`/`constrain_max` attributes.
  `initialize` collects the parameters that have either; a missing bound is unbounded. The
  bounds are kept as frozen parameters (so they are in the state_dict).
  """
  def __init__(self):
    """Initializes an empty constrainer (see `initialize`)."""
    super().__init__()
    # Must use lists since parameter names cannot contain '.' in `torch.nn.ParameterDict`.
    self.names: list[str] = []
    self.params = torch.nn.ParameterList()
    self.mins = torch.nn.ParameterList()
    self.maxes = torch.nn.ParameterList()

  def initialize(self, params: T.Iterable[tuple[str, torch.nn.Parameter]]):
    """Collects the constrained parameters among `params` (e.g. a model's `named_parameters()`)."""
    for name, param in params:
      if hasattr(param, 'constrain_min') or hasattr(param, 'constrain_max'):
        # A missing bound is unbounded, and a scalar bound applies to every element.
        min = getattr(param, 'constrain_min', torch.full_like(param, -torch.inf))
        max = getattr(param, 'constrain_max', torch.full_like(param, torch.inf))
        if min.shape != param.shape: min = min.expand(param.shape).clone()
        if max.shape != param.shape: max = max.expand(param.shape).clone()
        assert isinstance(min, torch.Tensor) and isinstance(max, torch.Tensor)
        assert min.shape == param.shape and max.shape == param.shape
        self.names.append(name)
        self.params.append(param)
        # Untrained parameters rather than non-persistent buffers, so that saved rollouts do not
        # store a copy per step and the bounds are in the state dict.
        self.mins.append(torch.nn.Parameter(min, requires_grad=False))
        self.maxes.append(torch.nn.Parameter(max, requires_grad=False))

  @torch.no_grad()
  def constrain(self):
    """Clamps every collected parameter into its bounds, in place. Call after every update."""
    for param, min, max in zip(self.params, self.mins, self.maxes):
      param[:] = torch.clamp(param, min, max)

  @T.override
  def extra_repr(self) -> str:
    return f'params=[\n{"" .join(
      [f"  {name}: {param.shape}\n" for name, param in zip(self.names, self.params)]
    )}]'

  def __repr__(self) -> str:
    return f'{self._get_name()}(\n  ' + self.extra_repr().replace('\n', '\n  ') + '\n)'


class DictModule(torch.nn.Module):
  """Applies one module to each of the given keys of a dict of tensors."""
  def __init__(self, module: T.Callable[[], torch.nn.Module], input_keys: T.Sequence[str]):
    """Initializes the module from its factory `module`, for the keys `input_keys`."""
    super().__init__()
    self.module = module()
    self.input_keys = input_keys

  def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Returns {key: module(input[key])} for the input keys."""
    return {k: self.module(input[k]) for k in self.input_keys}


def init_zeros(module: torch.nn.Module):
  """Zeroes the weight and bias of a `torch.nn.Linear` module (a tonic `init_fn`)."""
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.zeros_(module.weight)
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)


@torch.no_grad()
def constrain_sign(
  param: torch.nn.Parameter | torch.Tensor,
  magnitude: bool = False,
  zero: bool = False,
):
  """Applies sign constraints to a parameter based on its initial elements (see `constrain` with
  `sign=True`, which composes with other constraints).

  Args:
    param: Parameter to constrain.
    magnitude: If True, element magnitudes are constrained to be no smaller than their initial
      values.
      If False, element magnitudes are free and only signs are constrained.
    zero: If True, elements that are initially zero are constrained to remain zero. If False,
      elements that are initially zero are unconstrained.
  """
  x = param.data
  xmin = x.clone() if magnitude else torch.zeros(x.shape)
  xmax = x.clone() if magnitude else torch.zeros(x.shape)
  if zero:
    cmin = torch.where(x >= 0, xmin, -float('inf'))
    cmax = torch.where(x <= 0, xmax, +float('inf'))
  else:
    cmin = torch.where(x > 0, xmin, -float('inf'))
    cmax = torch.where(x < 0, xmax, +float('inf'))
  cmin.requires_grad = False
  cmax.requires_grad = False
  param.constrain_min = cmin  # type: ignore
  param.constrain_max = cmax  # type: ignore


@torch.no_grad()
def constrain_bound(
  param: torch.nn.Parameter | torch.Tensor,
  min: float | torch.Tensor | None = None,
  max: float | torch.Tensor | None = None,
):
  """Applies bound constraints to a parameter. A bound replaces any earlier bound.

  Args:
    param: Parameter to constrain.
    min: Minimum value or tensor of minimum values for the parameter. If None, no minimum
      constraint is applied.
    max: Maximum value or tensor of maximum values for the parameter. If None, no maximum
      constraint is applied.
  """
  if min is not None:
    param.constrain_min = torch.as_tensor(min, device=param.device)  # type: ignore
  if max is not None:
    param.constrain_max = torch.as_tensor(max, device=param.device)  # type: ignore


@torch.no_grad()
def constrain(
  param: torch.nn.Parameter | torch.Tensor,
  min: float | torch.Tensor | None = None,
  max: float | torch.Tensor | None = None,
  sign: bool = False,
  magnitude: bool = False,
  zero: bool = False,
  freeze: bool | T.Sequence[bool] | torch.Tensor = False,
):
  """
  Constrains a parameter or a view of a parameter. If given a view, the constraints will be applied
  to the corresponding elements of the base parameter.

  The constraints are elementwise bounds attached to the (base) parameter as `constrain_min` and
  `constrain_max`, which a `Constrainer` enforces. Repeated calls intersect the bounds, so
  constraints compose (e.g. a sign constraint on a whole weight and a freeze on a view of it).
  Elements' initial values are read when called.

  Args:
    param: Parameter, or a view of one (e.g. a block of a larger weight), to constrain.
    min: Lower bound (scalar or elementwise).
    max: Upper bound (scalar or elementwise).
    sign: Whether elements keep their initial sign: positive elements stay >= 0 (excitatory),
      negative elements stay <= 0 (inhibitory). Zero elements are free unless `zero`.
    magnitude: Whether nonzero elements keep at least their initial magnitude (with their sign).
    zero: Whether initially zero elements stay zero.
    freeze: Whether (elementwise, broadcast to the shape) elements stay at their initial values.
  """
  if min is None and max is None and not sign and not magnitude and not zero and freeze is False:
    return
  # Set up the constrain tensor fields on param or its base and get the corresponding views.
  base = param if param._base is None else param._base
  view = (lambda p: p) if param._base is None else functools.partial(
    torch.as_strided,
    size=param.shape,
    stride=param.stride(),
    storage_offset=param.storage_offset()
  )
  if not hasattr(base, 'constrain_min'):
    base.constrain_min = torch.full(base.shape, -float('inf'), device=base.device)  # type: ignore
  if not hasattr(base, 'constrain_max'):
    base.constrain_max = torch.full(base.shape, +float('inf'), device=base.device)  # type: ignore
  min_view: torch.Tensor = view(base.constrain_min)  # type: ignore
  max_view: torch.Tensor = view(base.constrain_max)  # type: ignore
  # Modify the elements of the constrain tensors based on the given constraints.
  x = param.data.clone()
  if min is not None:
    min_view[:] = torch.maximum(min_view, torch.as_tensor(min, device=param.device))
  if max is not None:
    max_view[:] = torch.minimum(max_view, torch.as_tensor(max, device=param.device))
  if sign:
    min_view[:] = torch.where(x > 0, torch.maximum(min_view, torch.zeros_like(x)), min_view)
    max_view[:] = torch.where(x < 0, torch.minimum(max_view, torch.zeros_like(x)), max_view)
  if magnitude:
    min_view[:] = torch.where(x > 0, torch.maximum(min_view, x), min_view)
    max_view[:] = torch.where(x < 0, torch.minimum(max_view, x), max_view)
  if zero:
    min_view[:] = torch.where(x == 0, 0, min_view)
    max_view[:] = torch.where(x == 0, 0, max_view)
  if freeze is not False:
    freeze = torch.as_tensor(freeze, device=param.device, dtype=torch.bool)
    freeze = freeze.broadcast_to(param.shape)
    min_view[:] = torch.where(freeze, x, min_view)
    max_view[:] = torch.where(freeze, x, max_view)
