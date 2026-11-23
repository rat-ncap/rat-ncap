import torch

from evolution2 import shaping

SCORES = torch.tensor([[3.0, -1.0, 10.0], [0.5, 2.0, -4.0]])


def test_identity_returns_the_scores():
  assert shaping.identity(SCORES) is SCORES


def test_scaled_rank_spaces_ranks_evenly_between_the_bounds():
  ranks = shaping.scaled_rank(torch.tensor([3.0, -1.0, 10.0, 0.5, 2.0]), lower=-1.0, upper=1.0)

  torch.testing.assert_close(ranks, torch.tensor([0.5, -1.0, 1.0, -0.5, 0.0]))


def test_scaled_rank_is_invariant_to_monotone_transforms():
  torch.testing.assert_close(shaping.scaled_rank(torch.exp(SCORES)), shaping.scaled_rank(SCORES))


def test_z_scores_normalize_over_all_elements_and_keep_the_shape():
  z = shaping.z_scores(SCORES)

  assert z.shape == SCORES.shape
  torch.testing.assert_close(z.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
  torch.testing.assert_close(z.std(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_modified_z_scores_center_on_the_median():
  z = shaping.modified_z_scores(torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0]))

  assert z[2] == 0
  assert z[4] > 10 * z[3]  # The outlier stands out.
