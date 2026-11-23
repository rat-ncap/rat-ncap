import time
import typing as T

import torch

from .. import core, noises, optimizers, shaping

__all__ = ['PointEvolutionStrategy']


class PointEvolutionStrategy(core.Algorithm):
  """A point-based evolution strategy maintains a single parent parameter vector (i.e. a point in
  parameter space), spawns multiple children by adding noise to the parent, and selects the next
  parent through a score-weighted average of children mutations (which can be viewed as a gradient
  estimate). Children mutations are evaluated in mirrored pairs to reduce the variance of the
  gradient estimate.

  Each train step:
  1. Spawns `population_size / 2` mutations `ε`, and the children `θ + σε` and `θ - σε`, where `θ`
     is the parent and `σ` is `mutation_scale`.
  2. Scores the children with the objective. All children of a step share one objective seed,
     which is also the mutation seed, derived from (seed, epoch): the pairs are compared under the
     same conditions, and a resumed run recomputes the same step.
  3. Keeps the `top_best` mutations whose pair has the highest score (the max of its two children),
     shapes the kept pairs' scores together, and estimates the gradient
     `g = gradient_scale / top_best * Σ (shaped⁺ - shaped⁻) ε`.
  4. Updates the parent to `θ + optimizer.step(g) - parameter_decay * θ`, then lets the objective
     modify it (`Objective.get_updated_params`).

  Each test step scores `test_size` copies of the parent with different objective seeds, derived
  from (seed, epoch, trial). These seeds have 12 digits (see `noises.multiseed`), so an objective
  that converts them to 32-bit integers (e.g. JAX without 64-bit mode) receives them wrapped around,
  which keeps them deterministic and distinct within a test step.

  The train stats and files `scores/raw` and `scores/shaped` hold only the kept pairs' scores,
  `[2 = {pos, neg}, top_best]`, not the whole population's.
  """
  def __init__(
    self,
    population_size: int = 64,
    test_size: int = 10,
    top_best: int | None = None,
    mutation_noise: T.Callable[[T.Sequence[int], int], torch.Tensor] = noises.gaussian,
    mutation_scale: float | torch.Tensor = 1.,
    score_shaping: T.Callable[[torch.Tensor], torch.Tensor] = shaping.identity,
    gradient_scale: float | torch.Tensor = 1.,
    parameter_decay: float = 0.,
    optimizer: optimizers.Optimizer | None = None,
  ):
    """Initializes the algorithm.

    Args:
      population_size: Number of children per train step, in mirrored pairs. Must be even.
      test_size: Number of times the parent is scored per test step.
      top_best: Number of mutations (pairs) kept to estimate the gradient, at most
        `population_size / 2`. `None` or 0 keeps all of them.
      mutation_noise: Function of `(size, seed)` that returns a noise block of that size.
      mutation_scale: Scale `σ` of the mutations.
      score_shaping: Function that shapes the kept pairs' scores, `[2, top_best]`.
      gradient_scale: Factor of the gradient estimate.
      parameter_decay: Factor of the parent subtracted from each update (weight decay).
      optimizer: Optimizer that turns the gradient estimate into an update. Defaults to `SGD()`.
    """
    assert population_size % 2 == 0, 'Population size must be even for mirrored sampling.'
    assert not top_best or top_best <= population_size // 2
    self.population_size = population_size
    self.test_size = test_size
    self.top_best = top_best or population_size // 2
    self.mutation_noise = mutation_noise
    self.mutation_scale = mutation_scale
    self.score_shaping = score_shaping
    self.gradient_scale = gradient_scale
    self.parameter_decay = parameter_decay
    self.optimizer = optimizer or optimizers.SGD()

  @T.override
  def initialize(self, objective: core.Objective, seed: int) -> torch.Tensor:
    """Initializes the objective and the parent from `objective.get_initial_params`, and returns
    the parent, `[1, param_size]`."""
    self.objective = objective
    self.objective.initialize(seed, self.population_size, self.test_size)
    seeds = torch.tensor([seed], dtype=torch.int64)
    self.param = self.objective.get_initial_params(seeds)  # Only 1 param point, [1, param_size].
    assert self.param.ndim == 2
    assert self.param.size(0) == 1
    self.param_size = self.param.size(1)
    self.optimizer.initialize(self.param)
    return self.param

  @T.override
  def step(self, epoch: int, seed: int) -> core.AlgorithmTrainData:
    """Spawns, scores and selects one generation, and updates the parent (see the class)."""
    # Spawn
    spawn_time = time.perf_counter()
    seed_epoch = noises.multiseed((3, 5), (seed, epoch))
    mutation_seed = seed_epoch
    mutation = self.mutation_noise((self.population_size // 2, self.param_size), mutation_seed)
    # Use mirrored sampling to evaluate 2 children per mutation vector.
    children = torch.cat((
      self.param + self.mutation_scale * mutation,
      self.param - self.mutation_scale * mutation,
    ))  # [batch_size = population_size, param_size]
    objective_seed = torch.full(
      (self.population_size,),
      seed_epoch,
      dtype=torch.int64,
    )  # [batch_size]
    spawn_time = time.perf_counter() - spawn_time

    # Score
    score_time = time.perf_counter()
    stats: dict[str, float | torch.Tensor] = {}
    files: dict[str, T.Any] = {}
    scores = self.objective.get_score(
      children,
      objective_seed,
      mode='train',
      epoch=epoch,
      stats=stats,
      files=files,
    )  # [population_size]
    best_param = children[scores.argmax()].clone()  # [param_size]
    scores = scores.reshape(2, self.population_size // 2)  # [2 = {pos, neg}, population_size / 2]
    del children  # Free memory
    score_time = time.perf_counter() - score_time

    # Select
    select_time = time.perf_counter()
    scores_pair = scores.max(dim=0).values  # Pair score is max of {pos, neg} [population_size / 2]
    indices_best = scores_pair.topk(self.top_best).indices  # [top_best]
    mutation = mutation[indices_best]  # [top_best, param_size]
    scores = scores[:, indices_best]  # [2, top_best]
    scores_shaped = self.score_shaping(scores)  # [2, top_best]
    scores_diff = scores_shaped[0, :] - scores_shaped[1, :]  # [top_best], pos - neg

    grad = scores_diff @ mutation  # [param_size] = [top_best] @ [top_best, param_size]
    grad = grad * self.gradient_scale / self.top_best
    update = self.optimizer.step(grad) - self.parameter_decay * self.param
    select_time = time.perf_counter() - select_time

    self.param = self.objective.get_updated_params(self.param + update)
    assert self.param.shape == (1, self.param_size)

    return core.AlgorithmTrainData(
      params=self.param,  # [1, param_size]
      best_param=best_param,  # [param_size]
      stats={
        'scores/raw': scores,
        'scores/shaped': scores_shaped,
        'spawn_time': spawn_time,
        'score_time': score_time,
        'select_time': select_time,
      } | stats,
      files={
        'scores/raw': scores,
        'scores/shaped': scores_shaped,
        'scores/diff': scores_diff,
        'grad': grad,
        'update': update,
      } | files,
    )

  @T.override
  def test(self, epoch: int, seed: int) -> core.AlgorithmTestData:
    """Scores `test_size` copies of the parent with different objective seeds."""
    objective_seed = torch.tensor(
      [noises.multiseed((3, 5, 3), (seed, epoch, trial)) for trial in range(self.test_size)],
      dtype=torch.int64
    )  # [test_size]
    # Must clone param to avoid issues with in-place ops mutating/constraining shared memory.
    param = self.param.broadcast_to(
      (self.test_size, self.param_size)
    ).clone()  # [test_size, param_size]
    stats: dict[str, float | torch.Tensor] = {}
    files: dict[str, T.Any] = {}
    scores = self.objective.get_score(
      param,
      objective_seed,
      mode='test',
      epoch=epoch,
      stats=stats,
      files=files,
    )
    return core.AlgorithmTestData(
      scores=scores,
      stats={'scores/raw': scores} | stats,
      files=files,
    )

  @T.override
  def get_state(self) -> dict[str, T.Any]:
    """Returns the parent and the optimizer state."""
    return {'param': self.param.cpu(), 'optimizer': self.optimizer.get_state()}

  @T.override
  def set_state(self, state: dict[str, T.Any]) -> None:
    """Restores the parent and the optimizer state from `get_state`."""
    self.param = state['param'].to(self.param)
    self.optimizer.set_state(state['optimizer'])
