import pytest
import torch

from utils import io as utio


@pytest.mark.parametrize(
  'filename, data',
  [
    ('state.pkl', dict(params=torch.arange(3.), epoch=2)),
    ('config.yaml', dict(b=1, a=[1, 2], c=dict(d='e'))),
    ('notes.txt', 'line 1\nline 2\n'),
    ('blob.bin', b'\x00\x01\xff'),
  ],
  ids=['pkl', 'yaml', 'txt', 'bin'],
)
def test_written_file_reads_back_equal(tmp_path, filename, data):
  path = tmp_path / 'missing' / 'directory' / filename

  utio.write_file(path, data)
  loaded = utio.read_file(path)

  if filename.endswith('.pkl'):
    torch.testing.assert_close(loaded, data)
  else:
    assert loaded == data
  if filename.endswith('.yaml'):
    assert list(loaded) == list(data)  # Key order is kept.


def test_lazy_read_loads_on_access(tmp_path):
  path = tmp_path / 'notes.txt'
  utio.write_file(path, 'text')

  ref = utio.read_file(path, lazy=True)

  assert isinstance(ref, utio.FileRef) and ref.data == 'text'


def test_loader_overrides_the_suffix(tmp_path):
  path = tmp_path / 'config.cfg'
  path.write_text('a: 1\n')

  assert utio.read_file(path, loader='.yaml') == dict(a=1)
