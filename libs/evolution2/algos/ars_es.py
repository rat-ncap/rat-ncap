import typing as T

import torch

from .. import optimizers, shaping
from . import point_es

__all__ = ['AugmentedRandomSearchEvolutionStrategy']


class AugmentedRandomSearchEvolutionStrategy(point_es.PointEvolutionStrategy):
  """Augmented Random Search (ARS) [Mania et al. 2018] (https://arxiv.org/abs/1803.07055) is a
  point-based evolution strategy that uses z-score score shaping (i.e. normalizing rewards by the
  standard deviation of the population) and direct SGD optimization.

  The step size is `gradient_scale * mutation_scale`: the parent moves by
  `gradient_scale * mutation_scale / top_best * Σ (z⁺ - z⁻) ε`, where `z` are the z-scores of the
  kept pairs' scores (see `PointEvolutionStrategy`). The algorithm does not normalize observations;
  that is left to the objective.
  """
  def __init__(
    self,
    mutation_scale: float | torch.Tensor = 0.02,
    score_shaping: T.Callable[[torch.Tensor], torch.Tensor] = shaping.z_scores,
    gradient_scale: float | torch.Tensor = 1.,
    optimizer: optimizers.Optimizer | None = None,
    **kwargs,
  ):
    """Initializes the algorithm.

    Args:
      mutation_scale: Scale `σ` of the mutations.
      score_shaping: Function that shapes the kept pairs' scores.
      gradient_scale: Step size relative to `mutation_scale`.
      optimizer: Optimizer of the gradient estimate. Defaults to `SGD(learning_rate=1)`.
      **kwargs: Other arguments of `PointEvolutionStrategy` (e.g. `population_size`, `top_best`).
    """
    super().__init__(
      mutation_scale=mutation_scale,
      gradient_scale=gradient_scale * mutation_scale,
      score_shaping=score_shaping,
      optimizer=optimizer or optimizers.SGD(learning_rate=1),
      **kwargs,
    )
