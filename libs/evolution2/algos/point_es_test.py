import typing as T

import torch

import evolution2
from evolution2 import algos, noises


class LinearObjective(evolution2.Objective):
  """Scores parameter vectors by their dot product with `weight`, and records the train children."""

  def __init__(self, weight: torch.Tensor, initial: torch.Tensor | None = None):
    self.weight = weight
    self.initial = initial if initial is not None else torch.zeros(len(weight))
    self.children: list[torch.Tensor] = []

  def get_initial_params(self, seeds: torch.Tensor) -> torch.Tensor:
    return self.initial.unsqueeze(0)

  def get_score(self, params: torch.Tensor, seeds: torch.Tensor, **kwargs) -> torch.Tensor:
    if kwargs['mode'] == 'train': self.children.append(params.clone())
    return params @ self.weight


class QuadraticObjective(evolution2.Objective):
  """Scores parameter vectors by their negative squared distance to a fixed target."""

  def get_initial_params(self, seeds: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, 4)

  def get_score(self, params: torch.Tensor, seeds: torch.Tensor, **kwargs) -> torch.Tensor:
    return -((params - torch.arange(4.)) ** 2).sum(dim=1)


def fixed_noise(mutation: torch.Tensor) -> T.Callable[[T.Sequence[int], int], torch.Tensor]:
  """A `mutation_noise` that always returns `mutation`."""
  def noise(size: T.Sequence[int], seed: int) -> torch.Tensor:
    assert tuple(size) == mutation.shape
    return mutation
  return noise


# ==================================================================================================
# PointEvolutionStrategy


def test_step_spawns_mirrored_children_reproducibly_per_seed_and_epoch():
  def spawn(epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the children and mutations of a fresh algorithm's step at `epoch`."""
    mutations = []
    def noise(size, seed):
      mutations.append(noises.gaussian(size, seed))
      return mutations[-1]
    objective = LinearObjective(torch.ones(3), initial=torch.tensor([1.0, 2.0, 3.0]))
    algorithm = algos.PointEvolutionStrategy(population_size=8, mutation_noise=noise)
    algorithm.initialize(objective, seed=0)
    algorithm.step(epoch, seed=0)
    return objective.children[0], mutations[0]

  children, mutation = spawn(epoch=1)
  again, _ = spawn(epoch=1)
  other, _ = spawn(epoch=2)

  parent = torch.tensor([1.0, 2.0, 3.0])
  mirrored = torch.cat((parent + mutation, parent - mutation))
  torch.testing.assert_close(children, mirrored, rtol=0, atol=0)
  torch.testing.assert_close(again, children, rtol=0, atol=0)
  assert not torch.equal(other, children)


def test_step_keeps_the_top_best_pairs_by_their_better_child():
  # Only the first coordinate is scored, so only the first mutation's pair differs in score.
  objective = LinearObjective(torch.tensor([1.0, 0.0, 0.0]))
  algorithm = algos.PointEvolutionStrategy(
    population_size=6,
    top_best=1,
    mutation_noise=fixed_noise(torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
    score_shaping=evolution2.shaping.z_scores,
  )
  algorithm.initialize(objective, seed=0)

  data = algorithm.step(1, seed=0)

  assert data.params[0, 0] > 0
  torch.testing.assert_close(data.params[0, 1:], torch.zeros(2), rtol=0, atol=0)
  assert data.files['scores/raw'].shape == (2, 1)  # Only the kept pair.


def test_ars_step_matches_the_hand_computed_update():
  weight = torch.tensor([1.0, -2.0])
  mutation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
  sigma, gradient_scale = 0.1, 2.0
  algorithm = algos.AugmentedRandomSearchEvolutionStrategy(
    population_size=6,
    top_best=2,
    mutation_scale=sigma,
    gradient_scale=gradient_scale,
    mutation_noise=fixed_noise(mutation),
  )
  algorithm.initialize(LinearObjective(weight), seed=0)

  data = algorithm.step(1, seed=0)

  # The children score ±σ w·ε = ±(0.1, -0.2, -0.05), so the pairs' better children score
  # (0.1, 0.2, 0.05), and the top 2 pairs are mutations 1 and 0, in that order.
  pos = sigma * mutation @ weight
  scores = torch.stack((pos, -pos))[:, [1, 0]]
  z = (scores - scores.mean()) / (scores.std() + 1e-6)
  expected = gradient_scale * sigma / 2 * (z[0] - z[1]) @ mutation[[1, 0]]
  torch.testing.assert_close(data.params[0], expected)


def test_ars_improves_the_score_of_a_quadratic_objective():
  algorithm = algos.AugmentedRandomSearchEvolutionStrategy(
    population_size=16,
    test_size=1,
    mutation_scale=0.1,
    gradient_scale=1.0,
  )
  algorithm.initialize(QuadraticObjective(), seed=0)
  initial = algorithm.test(0, seed=0).scores.item()

  for epoch in range(1, 21):
    algorithm.step(epoch, seed=0)

  assert algorithm.test(20, seed=0).scores.item() > initial / 2
