import collections
import logging
import os
import pathlib
import pprint
import typing as T

import numpy as np
import pandas as pd
import torch
import yaml

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
    self.dirpath = dirpath
    self._logger = logger

  @T.overload
  def save(self, first: str | pathlib.Path, second: T.Any) -> T.Self:
    ...

  @T.overload
  def save(self, first: T.Mapping[str, T.Any], second: None = None) -> T.Self:
    ...

  def save(self, first, second=None) -> T.Self:
    if isinstance(first, (str, pathlib.Path)):
      # Store single file.
      try:
        utils.io.write_file(self.dirpath / first, second)
      except Exception as e:
        self._logger.exception(f'Failed to save file: {first}')
        raise e
    elif isinstance(first, dict):
      # Store multiple files
      for filename, data in first.items():
        try:
          utils.io.write_file(self.dirpath / filename, data)
        except Exception as e:
          self._logger.exception(f'Failed to save file: {filename}')
          raise e
    else:
      raise TypeError(f'Unrecognized argument type: {first}')
    return self


YamlPrimitive = None | str | bool | int | float
YamlComposite = list['YamlPrimitive | YamlComposite'] | dict[str, 'YamlPrimitive | YamlComposite']
LedgerRecord = dict[str, YamlPrimitive | YamlComposite]


class LedgerSaver:
  # log.ledger('main').append(record)
  # log.ledger('main').extend([record1, record2])
  """Save a list of records to a YAML file, used with `Logger.ledger()`. The list of records is
  append-only. Each record is a dict. Keys must be strings. Values can be any YAML-serializable
  type: `None`, `str`, `bool`, `int`, `float`, `list`, `dict`.

  Append a single record:
    ```
    logger.ledger('filename.yaml').append({ 'key1': value1, 'key2': value2 })
    ```

  Append multiple records:
    ```
    logger.ledger('filename.yaml').extend([record1, record2])
    ```

  Load existing records from file:
    ```
    logger.ledger('filename.yaml').load()
    ```
  """
  def __init__(self, filepath: pathlib.Path, logger: logging.Logger):
    self.filepath = filepath
    self._logger = logger
    self._records: list[LedgerRecord] = []

  @property
  def data(self):
    return self._records

  def append(self, record: LedgerRecord) -> T.Self:
    try:
      records = [record]  # Convert to list so single record can be appended.
      with open(self.filepath, 'at') as file:
        # Assumes file is terminated by a newline. Too slow to check every time as that would
        # require read permissions and seeking to the end. Too unreadable to insert a newline
        # blindly before each record.
        yaml.dump(records, file, yaml.CSafeDumper, sort_keys=False)
      self._records.append(record)
    except Exception:
      self._logger.exception(f'Failed to append ledger file: {self.filepath}')
      raise
    return self

  def extend(self, records: T.Sequence[LedgerRecord]) -> T.Self:
    try:
      records = list(records)
      with open(self.filepath, 'at') as file:
        # Assumes file is terminated by a newline (see `append()`).
        yaml.dump(records, file, yaml.CSafeDumper, sort_keys=False)
      self._records.extend(records)
    except Exception:
      self._logger.exception(f'Failed to extend ledger file: {self.filepath}')
      raise
    return self

  def load(self) -> T.Self:
    if not self.filepath.exists(): return self
    try:
      with open(self.filepath, 'rt') as file:
        records = yaml.load(file, yaml.CSafeLoader)
        if records is not None:
          if not isinstance(records, list):
            raise TypeError(f'Invalid ledger file format: {self.filepath}')
          self._records = records
    except Exception:
      self._logger.exception(f'Failed to load ledger file: {self.filepath}')
      raise
    return self


KeyValueItem = YamlPrimitive | YamlComposite
KeyValueData = dict[str, KeyValueItem]


class KeyValueSaver:
  """Save key-value pairs to a YAML file, used with `Logger.keyval()`. Keys must be strings. Values
  can be any YAML-serializable type: `None`, `str`, `bool`, `int`, `float`, `list`, `dict`.

  Add a single key-value pair (or update existing):
    ```
    logger.keyval('filename.yaml').store('key', value)
    ```

  Add multiple key-value pairs (or update existing):
    ```
    logger.keyval('filename.yaml').store({ 'key1': value1, 'key2': value2 })
    ```

  Append to a key's existing values:
    ```
    logger.keyval('filename.yaml').append('key', value)
    ```

  Extend a key's existing values:
    ```
    logger.keyval('filename.yaml').extend('key', [value1, value2])
    ```

  Save to file (overwriting any existing content):
    ```
    logger.keyval('filename.yaml').save()
    ```
  """
  def __init__(self, filepath: pathlib.Path, logger: logging.Logger):
    self.filepath = filepath
    self._logger = logger
    self._data: KeyValueData = {}

  @property
  def data(self):
    return self._data

  @T.overload
  def store(self, first: str, second: KeyValueItem, *, prefix: str = '') -> T.Self:
    ...

  @T.overload
  def store(
    self, first: T.Mapping[str, KeyValueItem], second: None = None, *, prefix: str = ''
  ) -> T.Self:
    ...

  def store(self, first, second=None, *, prefix='') -> T.Self:
    if isinstance(first, str):
      # Store single key-value.
      self._data[f'{prefix}{first}'] = second
    elif isinstance(first, dict):
      # Store multiple keys-values.
      self._data.update({f'{prefix}{key}': value for key, value in first.items()})
    else:
      raise TypeError(f'Unrecognized argument type: {first}')
    return self

  def append(self, key: str, value: KeyValueItem) -> T.Self:
    if key in self._data:
      prev = self._data[key]
      if isinstance(prev, list):
        prev.append(value)
      else:
        raise TypeError(f'Appending value to non-list key: {key} <- {value}')
    else:
      self._data[key] = [value]
    return self

  def extend(self, key: str, values: T.Iterable[KeyValueItem]) -> T.Self:
    if key in self._data:
      prev = self._data[key]
      if isinstance(prev, list):
        prev.extend(values)
      else:
        raise TypeError(f'Extending values to non-list key: {key} <- {list(values)}')
    else:
      self._data[key] = list(values)
    return self

  def save(self) -> T.Self:
    try:
      with open(self.filepath, 'wt') as file:
        yaml.dump(self._data, file, yaml.CSafeDumper, sort_keys=False)
    except Exception as e:
      self._logger.exception(f'Failed to save key-value file: {self.filepath}')
      raise e
    return self

  def load(self) -> T.Self:
    if not self.filepath.exists(): return self
    try:
      with open(self.filepath, 'rt') as file:
        data = yaml.load(file, yaml.CSafeLoader)
        if data is not None: self._data = data
    except Exception as e:
      self._logger.exception(f'Failed to load key-value file: {self.filepath}')
      raise e
    return self


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
    self.filepath = filepath
    self._logger = logger
    self._active_row: dict[str, TableSingleOrSequenceValue] = {}
    self._unsaved_rows: list[dict[str, TableSingleValue]] = []
    self._saved_rows: list[dict[str, TableSingleValue]] = []
    self._columns: dict[str, T.Literal[True]] = {}  # Used as ordered set for quick lookup.
    self._overwrite: bool = True  # If columns change, need to overwrite file not just append.

  @property
  def data(self):
    return pd.DataFrame(self._saved_rows, columns=list(self._columns.keys()))

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
    if len(self._active_row) > 0: self.row()  # Finalize active row if non-empty.
    if len(self._unsaved_rows) == 0: return self
    if self._overwrite:
      # Columns changed, so need to overwrite entire file.
      all_rows = self._saved_rows + self._unsaved_rows
      df = pd.DataFrame(all_rows, columns=list(self._columns.keys()))
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
      df = pd.DataFrame(self._unsaved_rows, columns=list(self._columns.keys()))
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
      self._saved_rows = df.to_dict(orient='records')  # type: ignore
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
  and tabular data."""
  def __init__(
    self,
    log_dir: os.PathLike | str = './logs',
    *,
    name: str = '',
    level: LoggerLevel = 'notset',
    pretty: bool = False,
  ):
    self.log_dir = pathlib.Path(os.path.expandvars(os.path.expanduser(log_dir))).absolute()
    self.name = name or __name__  # By default, name using module where the logger is initialized.
    self.level = level
    self.pretty = pretty
    self._logger = logging.getLogger(self.name)
    self._logger.setLevel(_LEVELS[level])

    # Initialize stored savers.
    self._ledgers: dict[str, LedgerSaver] = {}
    self._keyvals: dict[str, KeyValueSaver] = {}
    self._tables: dict[str, TableSaver] = {}
    self._active_ledger: LedgerSaver | None = None
    self._active_keyval: KeyValueSaver | None = None
    self._active_table: TableSaver | None = None

  def __del__(self):
    self.finalize()

  def initialize(self):
    self.load_all()

  def finalize(self):
    self.save_all()

  def load_all(self):
    for ledger in self._ledgers.values():
      ledger.load()
    for keyval in self._keyvals.values():
      keyval.load()
    for table in self._tables.values():
      table.load()

  def save_all(self):
    # Files save on each operation (atomic write).
    # Ledger saves on each operation (atomic append).
    for keyval in self._keyvals.values():
      keyval.save()
    for table in self._tables.values():
      table.save()

  # ------------------------------------------------------------------------------------------------
  # File logging (atomic write)

  def file(self, *subdirpaths: str | os.PathLike):
    full_dirpath = pathlib.Path(self.log_dir, *subdirpaths)
    full_dirpath.mkdir(parents=True, exist_ok=True)
    return FileSaver(full_dirpath, self._logger)

  @property
  def fi(self):
    return self.file()

  # TODO: Add image (img), video (vid), plot (plt).

  # ------------------------------------------------------------------------------------------------
  # File logging (read + atomic append)

  def ledger(self, filepath: str = ''):
    # Empty filepath returns the active saver or creates a default saver.
    if not filepath and self._active_ledger is not None:
      return self._active_ledger
    if filepath:
      filepath = filepath + '.yaml' if not filepath.endswith('.yaml') else filepath
    else:
      filepath = 'ledger.yaml'

    # Non-empty filepath returns the existing saver or creates a new saver.
    if filepath in self._ledgers:
      self._active_ledger = self._ledgers[filepath]
    else:
      full_filepath = self.log_dir / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      saver = LedgerSaver(full_filepath, self._logger)
      self._ledgers[filepath] = saver
      self._active_ledger = saver
    return self._active_ledger

  @property
  def lgr(self):
    return self.ledger()

  @property
  def ledgers(self):
    return self._ledgers

  # ------------------------------------------------------------------------------------------------
  # File logging (read + incremental build-and-write)

  def keyval(self, filepath: str = ''):
    # Empty filepath returns the active saver or creates a default saver.
    if not filepath and self._active_keyval is not None:
      return self._active_keyval
    if filepath:
      filepath = filepath + '.yaml' if not filepath.endswith('.yaml') else filepath
    else:
      filepath = 'keyval.yaml'

    # Non-empty filepath returns the existing saver or creates a new saver.
    if filepath in self._keyvals:
      self._active_keyval = self._keyvals[filepath]
    else:
      full_filepath = self.log_dir / filepath
      full_filepath.parent.mkdir(parents=True, exist_ok=True)
      saver = KeyValueSaver(full_filepath, self._logger)
      self._keyvals[filepath] = saver
      self._active_keyval = saver
    return self._active_keyval

  @property
  def kv(self):
    return self.keyval()

  @property
  def keyvals(self):
    return self._keyvals

  def table(self, filepath: str = ''):
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
  def tab(self):
    return self.table()

  @property
  def tables(self):
    return self._tables

  # ------------------------------------------------------------------------------------------------
  # Message logging

  def debug(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.DEBUG): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.debug(sep.join(args), **kwargs)

  def info(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.INFO): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.info(sep.join(args), **kwargs)

  def warning(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.WARNING): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.warning(sep.join(args), **kwargs)

  def error(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.ERROR): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.error(sep.join(args), **kwargs)

  def critical(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.CRITICAL): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.critical(sep.join(args), **kwargs)

  def exception(self, *args: T.Any, pretty: bool = False, sep: str = ' ', **kwargs):
    if not self._logger.isEnabledFor(logging.ERROR): return
    args = tuple(
      pprint.pformat(arg, **_PFORMAT_KWS) if pretty or self.pretty else str(arg) for arg in args
    )
    self._logger.exception(sep.join(args), **kwargs)


class LoggingConfig(T.TypedDict):
  log_dir: os.PathLike | str
  name: str
  level: LoggerLevel
  pretty: bool


_GLOBAL_DEFAULTS: LoggingConfig = {
    'log_dir': './logs',
    'name': '',
    'level': 'notset',
    'pretty': False,
}
_LOGGERS: dict[str, Logger] = {}


def basicConfig(
  log_dir: os.PathLike | str | None = None,
  *,
  level: LoggerLevel | None = None,
  pretty: bool | None = None,
) -> None:
  if log_dir is not None:
    log_dir = pathlib.Path(os.path.expandvars(os.path.expanduser(log_dir))).absolute()
    _GLOBAL_DEFAULTS['log_dir'] = log_dir
  if level is not None:
    _GLOBAL_DEFAULTS['level'] = level
  if pretty is not None:
    _GLOBAL_DEFAULTS['pretty'] = pretty
  for logger in _LOGGERS.values():
    if log_dir is not None:
      logger.log_dir = log_dir
    if level is not None:
      logger.level = level
      logger._logger.setLevel(_LEVELS[level])
    if pretty is not None:
      logger.pretty = pretty


def getLogger(
  name: str | None = None,
) -> Logger:
  name = name or __name__
  if name in _LOGGERS:
    return _LOGGERS[name]
  config: LoggingConfig = _GLOBAL_DEFAULTS.copy()
  logger = Logger(**config)
  _LOGGERS[name] = logger
  return logger
