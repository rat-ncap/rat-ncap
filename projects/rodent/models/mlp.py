import typing as T

import torch

import tonic.torch.models as ttm

__all__ = ['init_zeros', 'mlp_es']


def init_zeros(module: torch.nn.Module):
  """Zeroes the weight and bias of a `torch.nn.Linear` module (a tonic `init_fn`)."""
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.zeros_(module.weight)
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)


def mlp_es(
  actor_hidden_sizes: T.Sequence[int] | None = (256, 256),
  actor_hidden_activation: T.Callable = torch.nn.ReLU,
  actor_hidden_init: T.Callable | None = init_zeros,
  actor_head_activation: T.Callable = torch.nn.Tanh,
  actor_head_bias: bool = True,
  actor_head_init: T.Callable | None = init_zeros,
  observation_normalizer: bool = True,
) -> ttm.ActorOnly:
  """Builds an MLP model for evolution strategies: an actor with a feedforward torso and a
  deterministic action head.

  The default inits zero the hidden and head layers. `None` keeps PyTorch's default init (the
  provided MLP configs set both to `None`).

  Args:
    actor_hidden_sizes: Sizes of the hidden layers, or `None` for no hidden layers.
    actor_hidden_activation: Activation of the hidden layers.
    actor_hidden_init: Initializer of the hidden layers (a tonic `init_fn`), or `None`.
    actor_head_activation: Activation of the head (e.g. `Tanh` for actions in [-1, 1]).
    actor_head_bias: Whether the head has a bias.
    actor_head_init: Initializer of the head (a tonic `init_fn`), or `None`.
    observation_normalizer: Whether the observations are rescaled to zero mean and unit standard
      deviation (statistics updated during training, not trained by the algorithm).

  Returns:
    The model.
  """
  return ttm.ActorOnly(
    actor=ttm.Actor(
      encoder=ttm.ObservationEncoder(),
      torso=ttm.MLP(
        sizes=actor_hidden_sizes,
        activation=actor_hidden_activation,
        init_fn=actor_hidden_init,
      ) if actor_hidden_sizes else None,
      head=ttm.DeterministicPolicyHead(
        activation=actor_head_activation,
        bias=actor_head_bias,
        init_fn=actor_head_init,
      ),
    ),
    observation_normalizer=ttm.MeanStdNormalizer() if observation_normalizer else None,
  )

