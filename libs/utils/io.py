import functools
import json
import os
import pathlib
import typing as T

import dill
import numpy as np
import pandas as pd
import yaml

CheckpointData = T.TypeVar('CheckpointData')


def dill_loader(folder: pathlib.Path) -> T.Any:
  with open(folder / 'data.pkl', 'rb') as file:
    return dill.load(file)


ParamSpec = T.ParamSpec("ParamSpec")
Return = T.TypeVar("Return")


FileData = T.Any
"""Contents of a file, which can be various data formats."""


class FileRef:
  """Reference to a file. The file data is lazily loaded when `data` is accessed, by the format of
  `loader`, a suffix (e.g. '.yaml'), or else of the file's suffix (see `read_file`)."""

  def __init__(
    self,
    path: pathlib.Path,
    *,
    loader: str | None = None,
    **kwargs,
  ):
    self.path = path
    self.loader = loader
    self.kwargs = kwargs

  @property
  def name(self):
    return self.path.name

  @functools.cached_property
  def data(self) -> FileData:
    match self.loader or self.path.suffix:
      case '.txt' | '.log':
        with open(self.path, 'rt') as file:
          return file.read()
      case '.yaml' | '.yml':
        with open(self.path, 'rt') as file:
          return yaml.load(file, yaml.CSafeLoader)
      case '.json':
        with open(self.path, 'rt') as file:
          return json.load(file, **self.kwargs)
      case '.csv':
        return pd.read_csv(self.path, **self.kwargs)
      case '.bin':
        with open(self.path, 'rb') as file:
          return file.read()
      case '.npy' | '.npz':
        return np.load(self.path, **self.kwargs)
      case '.pkl' | '' | _:
        with open(self.path, 'rb') as file:
          return dill.load(file, **self.kwargs)

  def __str__(self):
    return self.path.name

  def __repr__(self):
    return f"FileRef('{self.path.name}')"


@T.overload
def read_file(
  filepath: os.PathLike | str,
  *,
  lazy: T.Literal[False] = ...,
  loader: str | None = ...,
  **kwargs,
) -> FileData:
  ...


@T.overload
def read_file(
  filepath: os.PathLike | str,
  *,
  lazy: T.Literal[True] = ...,
  loader: str | None = ...,
  **kwargs,
) -> FileRef:
  ...


def read_file(filepath, *, lazy=False, loader=None, **kwargs):
  """Read a single file from disk. If `lazy=True`, a `FileRef` object is returned instead of the
  file data itself.

  The format follows `loader` or else the file's suffix: '.txt'/'.log' (text), '.yaml'/'.yml',
  '.json', '.csv' (`pd.read_csv`), '.bin' (bytes), '.npy'/'.npz', and any other suffix is unpickled
  with `dill`. Unpickling can run code: only read trusted files.
  """
  filepath = pathlib.Path(os.path.expandvars(os.path.expanduser(filepath)))
  ref = FileRef(filepath, loader=loader, **kwargs)
  return ref if lazy else ref.data


def write_file(
  filepath: os.PathLike | str,
  data: FileData,
  reload: bool = False,
) -> FileData:
  """Write a single file to disk, creating its directory, in the format of its suffix: '.txt'
  (text), '.yaml'/'.yml', '.csv' (`DataFrame.to_csv`, with its index), '.bin' (bytes), and any
  other suffix is pickled with `dill` ('.json', '.npy' and '.npz' are not implemented). The file is
  written in place, not atomically. Returns the data, or the file read back if `reload`."""
  filepath = pathlib.Path(os.path.expandvars(os.path.expanduser(filepath)))
  filepath.parent.mkdir(parents=True, exist_ok=True)

  match filepath.suffix:
    case '.txt':
      with open(filepath, 'wt') as file:
        file.write(data)
    case '.yaml' | '.yml':
      with open(filepath, 'wt') as file:
        yaml.dump(data, file, yaml.CSafeDumper, sort_keys=False)
    case '.json':
      raise NotImplementedError()
    case '.csv':
      pd.DataFrame(data, copy=False).to_csv(filepath)
    case '.bin':
      with open(filepath, 'wb') as file:
        file.write(data)
    case '.npy':
      raise NotImplementedError()
    case '.npz':
      raise NotImplementedError()
    case '.pkl' | '' | _:
      with open(filepath, 'wb') as file:
        dill.dump(data, file)

  return read_file(filepath, lazy=False) if reload else data
