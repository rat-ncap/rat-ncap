import typing as T

import utils.string

K = T.TypeVar('K')
V = T.TypeVar('V')


class AttrDict(dict[str, T.Any]):
  """A dictionary that allows access to its keys as attributes."""
  def __init__(self, *args, **kwargs):
    """Initializes the dictionary as `dict`."""
    super().__init__(*args, **kwargs)
    self.__dict__ = self  # Attributes are the items (a reference cycle, freed by the collector).


class GlobDict(dict[str, V]):
  """A dictionary mapping glob patterns (`fnmatch`) to values. Using specific (non-pattern) keys
  will match glob patterns for key membership and value access.

  A key takes the value of the first pattern that matches it, in insertion order: an exact pattern
  does not take precedence over an earlier matching wildcard, so list specific patterns first.
  Membership (`in`) also expands `{...}` options and `[...]` ranges in the patterns (see
  `utils.string.in_glob_range`), while value access does not.

    ```
    >>> d = GlobDict({'a*': 1})
    >>> 'apple' in d
    True
    >>> d['apple']
    1
    >>> d['banana']
    Traceback (most recent call last):
      ...
    KeyError: 'banana'
    >>> d.get('banana', 2)
    2
    ```
  """
  def __contains__(self, key: str) -> bool:
    """Returns whether `key` is a pattern or matches a pattern after range expansion."""
    return super().__contains__(key) or utils.string.in_glob_range(self.keys(), key)

  def __getitem__(self, key: str) -> V:
    """Returns the value of the first pattern that matches `key`.

    Raises:
      KeyError: If no pattern matches `key`.
    """
    return utils.string.select_by_glob_range(self, key)

  def get(self, key: str, default: V | None = None) -> V | None:
    """Returns the value of the first pattern that matches `key`, or `default` if none does (so
    `None` is returned both for no match and for a matching pattern of value `None`)."""
    return utils.string.select_by_glob_range(self, key, default)


__all__ = ['AttrDict', 'GlobDict']
