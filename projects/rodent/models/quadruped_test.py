import typing as T

import gymnasium
import numpy as np
import pytest
import torch

import circuits as cc
import composer.utils.gym as cxg
from models import quadruped

LIMBS = ('FL', 'FR', 'HL', 'HR')
ACTION_SIZES = {'F': 3, 'H': 2}  # Position setpoints per forelimb and hindlimb.
PREFIX = 'body/actuators_pos'


def make_pattern_formation(**kwargs) -> quadruped.QuadrupedPatternFormationNetwork:
  """A PF of one position action prefix, with small flexor/extensor weights of mixed signs."""
  return quadruped.QuadrupedPatternFormationNetwork(
    flx={'F': {PREFIX: [0.5, -0.5, 0.]}, 'H': {PREFIX: [0.3, -0.3]}},
    ext={'F': {PREFIX: [-0.5, 0.5, 0.2]}, 'H': {PREFIX: [-0.3, 0.3]}},
    action_prefixes=(PREFIX,),
    **kwargs,
  )


def make_afferent_feedback(**kwargs) -> quadruped.QuadrupedAfferentFeedbackNetwork:
  """An AF from each limb's tactile signal: contact inhibits the flexor and excites the extensor."""
  return quadruped.QuadrupedAfferentFeedbackNetwork(
    observation_prefixes=['body/tactile'],
    flx={'F': [0., -1.], 'H': [0., -1.]},  # Over the rectified [tactile-, tactile+].
    ext={'F': [0., 1.], 'H': [0., 1.]},
    **kwargs,
  )


def spaces() -> tuple[gymnasium.spaces.Dict, gymnasium.spaces.Dict]:
  """The observation space (speed command, tactile per limb) and action space (positions)."""
  observation_space = gymnasium.spaces.Dict({
    'target_vx': cxg.Box(-1., 1., shape=(1,)),
    **{f'body/tactile_limb{limb}': cxg.Box(-1., 1., shape=(1,)) for limb in LIMBS},
  })
  action_space = gymnasium.spaces.Dict({
    f'{PREFIX}_limb{limb}': cxg.Box(-1., 1., shape=(ACTION_SIZES[limb[0]],)) for limb in LIMBS
  })
  return observation_space, action_space


def make_actor() -> quadruped.QuadrupedNCAPActor:
  """A quadruped NCAP actor of the small PF and AF, initialized for the spaces."""
  actor = quadruped.QuadrupedNCAPActor(
    pattern_formation=make_pattern_formation(),
    rhythm_generation=quadruped.QuadrupedRhythmGenerationCircuit(),
    afferent_feedback=make_afferent_feedback(),
    brainstem_command=quadruped.QuadrupedBrainstemCommand(weight=0.5, bias=0.1),
    constrainer=quadruped.ncap.Constrainer(),
  )
  actor.initialize(*spaces())
  return actor


def command_signal(actor: quadruped.QuadrupedNCAPActor, name: str) -> cc.Signal:
  """A command signal (e.g. `freq_L`) of the actor's RG."""
  rhythm_generation = T.cast(quadruped.QuadrupedRhythmGenerationCircuit,
                             actor.sim.rhythm_generation)
  return T.cast(cc.Signal, rhythm_generation[name])


def observation(speed: float = 0.6, contact: float = 0.) -> dict[str, torch.Tensor]:
  """One sample's observation."""
  return {
    'target_vx': torch.tensor([speed]),
    **{f'body/tactile_limb{limb}': torch.tensor([contact]) for limb in LIMBS},
  }


# ==================================================================================================
# Pattern Formation and Afferent Feedback


def test_pattern_formation_shares_heads_between_left_and_right_limbs():
  pattern_formation = make_pattern_formation()
  pattern_formation.initialize(spaces()[1])
  inputs = {limb: torch.tensor([0.8, 0.1]) for limb in LIMBS}

  actions = pattern_formation(inputs)

  assert {key: value.shape for key, value in actions.items()} == {
    f'{PREFIX}_limb{limb}': (ACTION_SIZES[limb[0]],) for limb in LIMBS
  }
  torch.testing.assert_close(actions[f'{PREFIX}_limbFL'], actions[f'{PREFIX}_limbFR'])
  torch.testing.assert_close(actions[f'{PREFIX}_limbHL'], actions[f'{PREFIX}_limbHR'])


def test_constraints_keep_the_initial_weight_signs():
  actor = make_actor()
  heads = {side: T.cast(torch.nn.Linear, head)
           for side, head in actor.pattern_formation.net.heads.items()}
  initial = {side: head.weight.detach().clone() for side, head in heads.items()}
  generator = torch.Generator().manual_seed(0)

  with torch.no_grad():
    for head in heads.values():
      head.weight.add_(10 * torch.randn(head.weight.shape, generator=generator))
  T.cast(quadruped.ncap.Constrainer, actor.constrainer).constrain()

  for side, head in heads.items():
    positive, negative = initial[side] > 0, initial[side] < 0
    assert (head.weight[positive] >= 0).all() and (head.weight[negative] <= 0).all()
  zeros = torch.cat([head.weight[initial[side] == 0] for side, head in heads.items()])
  assert len(zeros) and (zeros != 0).any()  # Initially zero weights are free.


@pytest.mark.parametrize(
  'contact, expected',
  [(1., [-1., 1.]), (-1., [0., 0.]), (0., [0., 0.])],
  ids=['contact', 'negative', 'none'],
)
def test_afferent_feedback_maps_contact_to_flexor_inhibition_and_extensor_excitation(
  contact, expected,
):
  afferent_feedback = make_afferent_feedback(constrain=None)
  afferent_feedback.initialize(spaces()[0])

  outputs = afferent_feedback(observation(contact=contact))

  for limb in LIMBS:
    torch.testing.assert_close(outputs[limb], torch.tensor(expected))


# ==================================================================================================
# QuadrupedNCAPActor


def test_actor_writes_the_brainstem_command_into_every_command_signal():
  actor = make_actor()
  actor.reset()

  actor(observation(speed=0.6))

  for signal in ('freq_L', 'sync_L', 'freq_R', 'sync_R'):
    torch.testing.assert_close(command_signal(actor, signal).value, torch.tensor([0.5 * 0.6 + 0.1]))


def test_actor_returns_every_action_in_the_action_space_order():
  actor = make_actor()
  actor.reset()

  actions = actor(observation())

  assert list(actions) == list(spaces()[1])


def test_actor_runs_under_vmap():
  actor = make_actor()
  actor.reset()
  params: dict[str, torch.Tensor] = dict(actor.named_parameters())
  buffers = {k: v.clone().repeat(2, *[1] * v.ndim) for k, v in actor.named_buffers()}
  batch = {k: torch.stack([v, v]) for k, v in observation().items()}
  batch['body/tactile_limbFL'] = torch.tensor([[0.], [1.]])  # Only the second foot touches.

  def step(buffers, observation):
    """One functional step of the actor."""
    return torch.func.functional_call(actor, (params, buffers), args=(observation,))

  for _ in range(5):
    actions = torch.func.vmap(step, randomness='different')(buffers, batch)

  assert actions[f'{PREFIX}_limbFL'].shape == (2, ACTION_SIZES['F'])
  assert not torch.equal(buffers['sim.rhythm_generation.aff_flx_FL.value'][0],
                         buffers['sim.rhythm_generation.aff_flx_FL.value'][1])


# ==================================================================================================
# Rhythm Generation


def test_rhythm_generation_alternates_flexors_and_extensors():
  torch.manual_seed(0)
  sim = cc.Simulator(timestep=30)
  sim.rhythm_generation = rhythm_generation = quadruped.QuadrupedRhythmGenerationCircuit()
  sim.init()
  outputs = []
  for _ in range(300):
    for signal in ('freq_L', 'sync_L', 'freq_R', 'sync_R'):
      T.cast(cc.Signal, rhythm_generation[signal]).value[:] = 0.6
    sim.step()
    outputs.append(rhythm_generation.get_output()['FL'].numpy())
  flexor, extensor = np.array(outputs)[50:].T

  flexor_active = flexor > 0
  assert (flexor_active[1:] & ~flexor_active[:-1]).sum() >= 2  # Bursts.
  assert ((flexor > 0) & (extensor > 0.5)).mean() < 0.1  # Rarely co-active.
