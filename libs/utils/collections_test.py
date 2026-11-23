import hydra
import omegaconf as oc
import pytest

from utils import collections as utc


def test_first_matching_pattern_in_insertion_order_wins():
  globs = utc.GlobDict({'rgF>F': 'own', '*': 0, 'rgH>F': 'never'})

  assert globs['rgF>F'] == 'own'
  assert globs['rgH>F'] == 0  # The earlier wildcard takes precedence over the exact pattern.


def test_unmatched_key_raises_and_get_returns_the_default():
  globs = utc.GlobDict({'a*': 1})

  with pytest.raises(KeyError, match='banana'):
    globs['banana']
  assert globs.get('banana', 2) == 2
  assert globs.get('banana') is None


@pytest.mark.parametrize(
  'pattern, key',
  [('a*', 'apple'), ('{a,b}_x', 'b_x'), ('x_[1-12]', 'x_12')],
  ids=['wildcard', 'options', 'range'],
)
def test_membership_expands_options_and_ranges(pattern, key):
  globs = utc.GlobDict({pattern: 1})

  assert key in globs
  assert 'other' not in globs


def test_instantiation_keeps_the_pattern_order():
  config = oc.OmegaConf.create({
    '_target_': 'utils.collections.GlobDict',
    'b*': 1,
    '*': 0,
  })

  globs = hydra.utils.instantiate(config, _convert_='object')

  assert list(globs) == ['b*', '*']
  assert (globs['bob'], globs['alice']) == (1, 0)

