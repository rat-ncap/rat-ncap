import logging
import os
import pathlib
import pprint
import typing as T

import numpy as np
import pandas as pd

import utils.io


class FileSaver:
  """Save files to a directory, used with `Logger.file()`. See `utils.io.write_file` for allowed
  file types.

  Save a single file:
    ```
    logger.file('dirname').save('filename', data)
    ```

  Save multiple files at once:
    ```
    logger.file('dirname').save({ 'filename1': data1, 'filename2': data2 })
    ```
  """
  def __init__(self, dirpath: pathlib.Path, logger: logging.Logger):
    """Initializes the saver of files in `dirpath`, logging failures to `logger`."""
    self.dirpath = dirpath
    self._logger = logger

  @T.overload
  def save(self, first: str | pathlib.Path, second: T.Any) -> T.Self:
    ...

  @T.overload
  def save(self, first: T.Mapping[str, T.Any], second: None = None) -> T.Self:
    ...

  def save(self, first, second=None) -> T.Self:
    """Writes a file `first` of data `second`, or each file of a `{filename: data}` dict `first`,
    with `utils.io.write_file` (format by suffix).

    Raises:
      TypeError: If `first` is neither a filename nor a dict.
    """
    if isinstance(first, (str, pathlib.Path)):
      # Store single file.
      try:
        utils.io.write_file(self.dirpath / first, second)
      except Exception:
        self._logger.exception(f'Failed to save file: {first}')
        raise
    elif isinstance(first, dict):
      # Store multiple files
      for filename, data in first.items():
        try:
          utils.io.write_file(self.dirpath / filename, data)
        except Exception:
          self._logger.exception(f'Failed to save file: {filename}')
          raise
    else:
      raise TypeError(f'Unrecognized argument type: {first}')
    return self


YamlPrimitive = None | str | bool | int | float
YamlComposite = list['YamlPrimitive | YamlComposite'] | dict[str, 'YamlPrimitive | YamlComposite']
LedgerRecord = dict[str, YamlPrimitive | YamlComposite]


KeyValueItem = YamlPrimitive | YamlComposite
KeyValueData = dict[str, KeyValueItem]


TableSingleInput = str | int | float | None
TableSequenceInput = T.Iterable[str] | T.Iterable[int] | T.Iterable[float] | np.ndarray
TableSingleOrSequenceInput = TableSingleInput | TableSequenceInput
TableSingleValue = str | int | float | None
TableSequenceValue = list[str] | list[int] | list[float]
TableSingleOrSequenceValue = TableSingleValue | TableSequenceValue


class TableSaver:
  """Save tabular data to a CSV file, used with `Logger.table()`. Rows are built incrementally and
  saved to the file in batches. Column names must be strings. Values can be any CSV-serializable
  type: `str`, `int`, `float`, `None`. If a value is a sequence of `str`, it will be serialized
  as "elem1|elem2|elem3". If a value is a sequence of `int`/`float` or an `np.ndarray`, it will be
  serialized as summary statistics columns: ".../mean", ".../std", ".../min", ".../max", ".../size".

  Add a single column (or update existing):
    ```
    logger.table('filename.csv').store('column', value)
    ```

  Add multiple columns (or update existing):
    ```
    logger.table('filename.csv').store({ 'column1': value1, 'column2': value2 })
    ```

  Append to a column's existing values:
    ```
    logger.table('filename.csv').append('column', value)
    ```

  Extend a column's existing values:
    ```
    logger.table('filename.csv').extend('column', [value1, value2])
    ```

  Flush the active row (filling in missing columns with `None`) and start a new row:
    ```
    logger.table('filename.csv').row()
    ```

  Save to file (overwriting any existing content):
    ```
    logger.table('filename.csv').save()
    ```
  """
  def __init__(self, filepath: pathlib.Path, logger: logging.Logger):
    """Initializes the saver of the CSV file `filepath`, logging failures to `logger`."""
    self.filepath = filepath
    self._logger = logger
    self._active_row: dict[str, TableSingleOrSequenceValue] = {}
    self._unsaved_rows: list[dict[str, TableSingleValue]] = []
    self._saved_rows: list[dict[str, TableSingleValue]] = []
    self._columns: dict[str, T.Literal[True]] = {}  # Used as ordered set for quick lookup.
    self._overwrite: bool = True  # If columns change, need to overwrite file not just append.

  @property
  def data(self) -> pd.DataFrame:
    """The saved rows (not those stored since the last `save`)."""
    return pd.DataFrame(self._saved_rows, columns=pd.Index(list(self._columns)))

  @T.overload
  def store(self, first: str, second: TableSingleOrSequenceInput, *, prefix: str = '') -> T.Self:
    ...

  @T.overload
  def store(
    self,
    first: T.Mapping[str, TableSingleOrSequenceInput],
    second: None = None,
    *,
    prefix: str = ''
  ) -> T.Self:
    ...

  def store(self, first, second=None, *, prefix='') -> T.Self:
    """Sets column `first` of the active row to `second`, or each column of a `{column: value}`
    dict `first`, with column names prefixed by `prefix`. Values that are not exactly `int`,
    `float`, `str` or `None` are stored as sequences (e.g. `np.float32` raises).

    Raises:
      TypeError: If `first` is neither a column name nor a dict.
    """
    if isinstance(first, str):
      # Store single value.
      items = {first: second}
    elif isinstance(first, dict):
      # Store multiple values.
      items = first
    else:
      raise TypeError(f'Unrecognized argument type: {first}')

    for column, value in items.items():
      column = f'{prefix}{column}'
      if isinstance(value, (int, float, str)) or value is None:
        self._active_row[column] = value
      elif isinstance(value, np.ndarray):
        self._active_row[column] = value.flatten().tolist()
      else:
        self._active_row[column] = list(value)
    return self

  def append(self, column: str, value: str | int | float) -> T.Self:
    """Appends `value` to the sequence of `column` in the active row.

    Raises:
      TypeError: If the column holds a single value.
    """
    if column in self._active_row:
      prev = self._active_row[column]
      if isinstance(prev, list):
        prev.append(value)  # type: ignore
      else:
        raise TypeError(f'Appending value to non-list column: {column} <- {value}')
    else:
      self._active_row[column] = [value]  # type: ignore
    return self

  def extend(self, column: str, values: TableSequenceInput) -> T.Self:
    """Extends the sequence of `column` in the active row with `values`.

    Raises:
      TypeError: If the column holds a single value.
    """
    lst = values.flatten().tolist() if isinstance(values, np.ndarray) else list(values)
    if column in self._active_row:
      prev = self._active_row[column]
      if isinstance(prev, list):
        prev.extend(lst)  # type: ignore
      else:
        raise TypeError(f'Extending values to non-list column: {column} <- {lst}')
    else:
      self._active_row[column] = lst  # type: ignore
    return self

  def row(self) -> T.Self:
    """Ends the active row and starts a new one. Numeric sequences become `<column>/mean`, `/std`,
    `/min`, `/max` and `/size` columns, and other sequences (including empty ones) a '|'-joined
    string, so a column's empty and non-empty sequences give different columns."""
    row = {}
    for column, value in self._active_row.items():
      if isinstance(value, (str, int, float)) or value is None:
        row[column] = value
      elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], (int, float)):
          row[f'{column}/mean'] = np.mean(value).item()  # type: ignore
          row[f'{column}/std'] = np.std(value).item()  # type: ignore
          row[f'{column}/min'] = np.min(value).item()
          row[f'{column}/max'] = np.max(value).item()
          row[f'{column}/size'] = len(value)
        else:
          row[column] = '|'.join([str(x) for x in value])
    self._unsaved_rows.append(row)
    self._active_row = {}
    for column in row.keys():
      if column not in self._columns:
        self._columns[column] = True
        self._overwrite = True
    return self

  def save(self) -> T.Self:
    """Ends the active row and writes the unsaved rows: appended to the file, or the whole file
    rewritten if the columns changed (earlier rows get empty new columns)."""
    if len(self._active_row) > 0: self.row()  # Finalize active row if non-empty.
    if len(self._unsaved_rows) == 0: return self
    if self._overwrite:
      # Columns changed, so need to overwrite entire file.
      all_rows = self._saved_rows + self._unsaved_rows
      df = pd.DataFrame(all_rows, columns=pd.Index(list(self._columns)))
      try:
        df.to_csv(self.filepath, index=False)
      except Exception:
        self._logger.exception(f'Failed to save table file: {self.filepath}')
        raise
      else:
        self._saved_rows = all_rows
        self._unsaved_rows = []
        self._overwrite = False
    else:
      # Columns same, so append rows to existing file.
      df = pd.DataFrame(self._unsaved_rows, columns=pd.Index(list(self._columns)))
      try:
        df.to_csv(self.filepath, mode='a', header=False, index=False)
      except Exception:
        self._logger.exception(f'Failed to save table file: {self.filepath}')
        raise
      else:
        self._saved_rows.extend(self._unsaved_rows)
        self._unsaved_rows = []
    return self

  def load(self) -> T.Self:
    """Loads the rows of an existing file as saved rows, so that later rows are appended."""
    if not self.filepath.exists(): return self
    try:
      df = pd.read_csv(self.filepath, index_col=False)
    except pd.errors.EmptyDataError:
      return self
    except Exception:
      self._logger.exception(f'Failed to load table file: {self.filepath}')
      raise
    else:
      self._columns = {column: True for column in df.columns}
      self._saved_rows = df.to_dict(orient='records')
      self._overwrite = False
    return self


LoggerLevel = T.Literal['notset', 'debug', 'info', 'warning', 'error', 'critical']

_LEVELS: dict[LoggerLevel, int] = {
  'notset': logging.NOTSET,
  'debug': logging.DEBUG,
  'info': logging.INFO,
  'warning': logging.WARNING,
  'error': logging.ERROR,
  'critical': logging.CRITICAL,
}

_PFORMAT_KWS: dict[str, T.Any] = {
  'indent': 2,
  'width': 100,
  'sort_dicts': False,
}


class Logger:
  """A logger that can save messages at different levels as well as save files, key-value pairs,
  and tabular data.

  Messages go to a stdlib logger, without a handler of its own: where they appear depends on the
  logging configuration (e.g. Hydra's job logging).
  """
  def __init__(
    self,
    log_dir: os.PathLike | str = './logs',
    *,
    name: str = '',
    level: LoggerLevel = 'notset',
    pretty: bool = False,
  ):
    """Initializes the logger.

    Args:
      log_dir: Directory of the saved files (`~` and environment variables expanded).
      name: Name of the stdlib logger. Defaults to this module's name, which all `Logger`s without
        a name share (the last `level` set applies to all of them).
      level: Level of the stdlib logger.
      pretty: Whether messages are pretty-printed (`pprint`) by default.
    """
    self.log_dir = pathlib.Path(os.path.expandvars(os.path.expanduser(log_dir))).absolute()
    self.name = name or __name__
    self.level = level
    self.pretty = pretty
    self._logger = logging.getLogger(self.name)
    self._logger.setLevel(_LEVELS[level])

    # Initialize stored savers.
    self._tables: dict[str, TableSaver] = {}
    self._active_table: TableSaver | None = None

  def __del__(self):
    """Saves all savers when the logger is deleted (see `finalize`)."""
    self.finalize()

  def initialize(self):
    """Loads all savers' existing files."""
    self.load_all()

  def finalize(self):
    """Saves all savers' unsaved data."""
    self.save_all()

  def load_all(self):
    """Loads the tables' existing files."""
    for table in self._tables.values():
      table.load()

  def save_all(self):
    """Saves the tables (files save on each operation)."""
    for table in self._tables.values():
      table.save()

  # ------------------------------------------------------------------------------------------------
  # File logging (written on each save)

  def file(self, *subdirpaths: str | os.PathLike) -> FileSaver:
    """Returns a saver of files in the log directory's subdirectory `subdirpaths`, created if
    needed."""
    full_dirpath = pathlib.Path(self.log_dir, *subdirpaths)
    full_dirpath.mkdir(parents=True, exist_ok=True)
    return FileSaver(full_dirpath, self._logger)

  # ------------------------------------------------------------------------------------------------
  # Table logging (written on each save)

  def table(self, filepath: str = '') -> TableSaver:
    """Returns the saver of the CSV file `filepath` (a '.csv' suffix added if missing) in the log
    directory, created on first use and then the active table; or, if `filepath` is empty, the
    active table (or 'table.csv')."""
    # Empty filepath returns the active saver or creates a default saver.
    if not filepath and self._active_table is not None:
      return self._active_table
    if filepath:
      filepath = filepath + '.csv' if not filepath.endswith('.csv') else filepath
    else:
      filepath = 'table.csv'

    # Non-empty filepath returns the existing saver or creates a new saver.
    if filepath in self._tables:
      self._active_table = self._tables[filepath]
    else:
      full_filepath = self.log_dir / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      saver = TableSaver(full_filepath, self._logger)
      self._tables[filepath] = saver
      self._active_table = saver
    return self._active_table

  @property
  def tables(self):
    return self._tables

  # ------------------------------------------------------------------------------------------------
  # Message logging

  def debug(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level
    'debug'."""
    if not self._logger.isEnabledFor(logging.DEBUG): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.debug(sep.join(args), **kwargs)

  def info(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level
    'info'."""
    if not self._logger.isEnabledFor(logging.INFO): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.info(sep.join(args), **kwargs)

  def warning(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level
    'warning'."""
    if not self._logger.isEnabledFor(logging.WARNING): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.warning(sep.join(args), **kwargs)

  def error(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level
    'error'."""
    if not self._logger.isEnabledFor(logging.ERROR): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.error(sep.join(args), **kwargs)

  def critical(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level
    'critical'."""
    if not self._logger.isEnabledFor(logging.CRITICAL): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.critical(sep.join(args), **kwargs)

  def exception(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    """Logs the arguments, joined by `sep` (each pretty-printed if `pretty`), at level 'error',
    with the exception being handled."""
    if not self._logger.isEnabledFor(logging.ERROR): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.exception(sep.join(args), **kwargs)
