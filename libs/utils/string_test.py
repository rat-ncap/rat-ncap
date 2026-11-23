import pytest

from utils import string as uts


@pytest.mark.parametrize(
  'spec, expected',
  [
    ('{foo,bar}_[1-2]', ['foo_1', 'foo_2', 'bar_1', 'bar_2']),
    ('x_[8-10]', ['x_8', 'x_9', 'x_10']),
    ('[a-c]', ['a', 'b', 'c']),
    ('plain_*', ['plain_*']),
  ],
  ids=['options_and_ints', 'multi_digit', 'chars', 'no_range'],
)
def test_glob_range_expands_options_and_ranges_in_order(spec, expected):
  assert list(uts.glob_range(spec)) == expected


@pytest.mark.parametrize(
  'item, matches',
  [('b_foof_1', True), ('c_foo_3', True), ('d_foo_1', False), ('b_bar_1', False)],
)
def test_in_glob_range_matches_wildcards_after_expansion(item, matches):
  assert uts.in_glob_range('[a-c]_foo*_[1-3]', item) == matches


def test_select_by_glob_range_returns_the_first_match_or_the_default():
  spec = dict(a_x='exact', a_y='other')

  assert uts.select_by_glob_range({'a_*': 1, 'a_x': 2}, 'a_x') == 1
  assert uts.select_by_glob_range(spec, 'a_x') == 'exact'
  assert uts.select_by_glob_range(spec, 'b', default=None) is None
  with pytest.raises(KeyError):
    uts.select_by_glob_range(spec, 'b')


def test_empty_char_span_is_rejected():
  with pytest.raises(ValueError, match='char span'):
    list(uts.char_range('a,,b'))


@pytest.mark.parametrize('parts', [1, 2, 5])
def test_hruid_has_the_given_number_of_parts(parts):
  assert len(uts.hruid(parts).split('-')) == parts
