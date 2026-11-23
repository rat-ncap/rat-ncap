import pytest
import torch

from evolution2 import noises


def test_multiseed_packs_each_seed_into_its_digits():
  assert noises.multiseed((2, 3, 4), (2, 333, 10004)) == 1023330004


def test_multiseed_wraps_seeds_longer_than_their_digits():
  assert noises.multiseed((3, 5), (0, 100000)) == noises.multiseed((3, 5), (0, 0))


def test_gaussian_reproduces_noise_of_the_same_seed():
  first = noises.gaussian((3, 4), 7)
  second = noises.gaussian((3, 4), 7)
  other = noises.gaussian((3, 4), 8)

  torch.testing.assert_close(first, second, rtol=0, atol=0)
  assert not torch.equal(first, other)


def test_gaussian_has_zero_mean_and_unit_std():
  noise = noises.gaussian((100_000,), 0)

  assert abs(noise.mean().item()) < 0.02
  assert abs(noise.std().item() - 1) < 0.02


def test_bernoulli_draws_minus_one_and_plus_one():
  noise = noises.bernoulli((1000,), 0)

  assert noise.unique().tolist() == [-1, 1]


@pytest.mark.parametrize('seed', [0, 1])
def test_uniform_stays_within_minus_one_and_plus_one(seed):
  noise = noises.uniform((1000,), seed)

  assert noise.min() >= -1 and noise.max() <= 1
