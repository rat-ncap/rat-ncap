import pathlib

import numpy as np
import torch

from utils import io as utio
from utils import logging as utlog


def read_lines(path: pathlib.Path) -> list[str]:
  """Returns the lines of a text file."""
  return path.read_text().splitlines()


def test_table_writes_scalar_and_sequence_summary_columns(tmp_path):
  logger = utlog.Logger(tmp_path)

  logger.table('train.csv').store(dict(a=1, b=np.array([1., 3.]))).save()

  assert read_lines(tmp_path / 'train.csv') == [
    'a,b/mean,b/std,b/min,b/max,b/size',
    '1,2.0,1.0,1.0,3.0,2',
  ]


def test_loaded_table_appends_rows_under_one_header(tmp_path):
  first = utlog.Logger(tmp_path)
  for step in range(2):
    first.table('train.csv').store(dict(step=step)).save()
  resumed = utlog.Logger(tmp_path)

  resumed.table('train.csv').load().store(dict(step=2)).save()

  assert read_lines(tmp_path / 'train.csv') == ['step', '0', '1', '2']


def test_new_column_rewrites_the_table_with_empty_earlier_values(tmp_path):
  logger = utlog.Logger(tmp_path)
  logger.table('train.csv').store(dict(a=1)).save()

  logger.table('train.csv').store(dict(a=2, b='x')).save()

  assert read_lines(tmp_path / 'train.csv') == ['a,b', '1,', '2,x']


def test_saved_file_reads_back_equal(tmp_path):
  logger = utlog.Logger(tmp_path)
  params = torch.arange(4.)

  logger.file('params').save('0.pkl', params)

  torch.testing.assert_close(utio.read_file(tmp_path / 'params' / '0.pkl'), params)
