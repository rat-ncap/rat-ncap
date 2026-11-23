import gymnasium
import pytest
import torch

import composer.utils.gym as cxg
import composer.utils.torch as cxt
from models import mlp

OBSERVATION_SIZES = {'joint_angles': 5, 'sensors': 7}
ACTION_SIZE = 11


def make_model(**kwargs):
  """An `mlp_es` model initialized for 5 + 7 observations and 11 actions."""
  model = mlp.mlp_es(**kwargs)
  model.initialize(
    gymnasium.spaces.Dict({k: cxg.Box(-1., 1., shape=(n,)) for k, n in OBSERVATION_SIZES.items()}),
    gymnasium.spaces.Dict({'forces': cxg.Box(-1., 1., shape=(ACTION_SIZE,))}),
  )
  return model


def observations(seed: int = 0) -> dict[str, torch.Tensor]:
  """A batch of 3 random observations."""
  generator = torch.Generator().manual_seed(seed)
  return {k: torch.randn(3, n, generator=generator) for k, n in OBSERVATION_SIZES.items()}


def actor_parameters(model) -> dict[str, torch.Tensor]:
  """The actor's trainable parameters, by name."""
  return {k: v for k, v in model.actor.named_parameters() if v.requires_grad}


# ==================================================================================================
# Initializers


def test_init_zeros_zeroes_a_linear_layer():
  layer = torch.nn.Linear(3, 2)

  mlp.init_zeros(layer)

  assert not layer.weight.any() and not layer.bias.any()


# ==================================================================================================
# mlp_es


def test_actor_parameters_match_the_layer_sizes():
  model = make_model(actor_hidden_sizes=(256, 256))

  count = sum(p.numel() for p in actor_parameters(model).values())

  inputs = sum(OBSERVATION_SIZES.values())
  assert count == inputs * 256 + 256 + 256 * 256 + 256 + 256 * ACTION_SIZE + ACTION_SIZE
  assert model.observation_normalizer is not None
  assert not any(p.requires_grad for p in model.observation_normalizer.parameters())


@pytest.mark.parametrize(
  'kwargs, zero',
  [({}, True), ({'actor_hidden_init': None, 'actor_head_init': None}, False)],
  ids=['zero_init', 'default_init'],
)
def test_actor_output_is_zero_only_with_zero_inits(kwargs, zero):
  model = make_model(**kwargs)

  action = model.actor(observations())['forces']

  assert bool((action == 0).all()) == zero


def test_packed_parameters_restore_the_actor_output():
  source = make_model(actor_hidden_init=None, actor_head_init=None)
  target = make_model()
  vector = cxt.pack_tensors(actor_parameters(source), grad_only=True)

  cxt.unpack_tensors(vector, actor_parameters(target), grad_only=True)

  torch.testing.assert_close(target.actor(observations()), source.actor(observations()))
  torch.testing.assert_close(cxt.pack_tensors(actor_parameters(target), grad_only=True), vector)
