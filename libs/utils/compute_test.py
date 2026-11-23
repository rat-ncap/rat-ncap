import subprocess

import pytest

from utils import compute as utcomp


@pytest.mark.parametrize(
  'error',
  [FileNotFoundError('nvidia-smi'), subprocess.CalledProcessError(9, 'nvidia-smi')],
  ids=['missing', 'failing'],
)
def test_gpuinfo_reports_no_gpus_without_a_working_nvidia_smi(monkeypatch, error):
  def check_output(*args, **kwargs):
    raise error

  monkeypatch.setattr(subprocess, 'check_output', check_output)

  assert utcomp.gpuinfo() == []


def test_gpuinfo_parses_each_gpu(monkeypatch):
  output = '0, NVIDIA L40S, 46068\n1, NVIDIA L40S, 46068\n'
  monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: output)

  gpus = utcomp.gpuinfo()

  assert gpus == [
    dict(index=0, model='NVIDIA L40S', memory_total=46068),
    dict(index=1, model='NVIDIA L40S', memory_total=46068),
  ]


def test_jobinfo_reports_the_process_and_its_cpus(monkeypatch):
  monkeypatch.setattr(utcomp, 'gpuinfo', lambda: [])

  info = utcomp.jobinfo()

  assert info['cpu_count'] == len(info['cpu_index']) >= 1
  assert info['gpu_count'] == 0 and info['gpu_model'] is None
