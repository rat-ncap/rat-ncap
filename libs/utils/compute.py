import datetime
import os
import platform
import socket
import subprocess
import typing as T

import psutil


class GpuInfo(T.TypedDict):
  """Static GPU information: `memory_total` in MiB (as nvidia-smi reports it)."""
  index: int
  model: str
  memory_total: int


def gpuinfo() -> list[GpuInfo]:
  """Get static GPU information (index, name, total memory), or no GPUs if `nvidia-smi` is
  unavailable or fails."""
  command = ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits']
  try:
    info = subprocess.check_output(command, encoding='utf-8').strip().split('\n')
    gpus: list[GpuInfo] = []
    for line in info:
      index, model, mem_total = [x.strip() for x in line.split(',')]
      gpus.append({
        'index': int(index),
        'model': model,
        'memory_total': int(mem_total),
      })
    return gpus
  except (OSError, subprocess.CalledProcessError, ValueError):
    return []


class JobInfo(T.TypedDict):
  """Static job information: `cpu_count`/`cpu_index` are the CPUs the process may run on (e.g. the
  job's allocation), `creation` an ISO timestamp."""
  pid: int
  name: str
  host: str
  platform: str
  release: str
  cpu_count: int
  cpu_index: list[int]
  gpu_count: int
  gpu_index: list[int]
  gpu_model: str | None
  threads: int
  creation: str
  cwd: str


def jobinfo() -> JobInfo:
  """Get static job information. Linux only (`os.sched_getaffinity`)."""
  process = psutil.Process(os.getpid())
  cpus = list(os.sched_getaffinity(0))
  gpus = gpuinfo()
  return {
    'pid': process.pid,
    'name': process.name(),
    'host': socket.gethostname(),
    'platform': platform.system(),
    'release': platform.release(),
    'cpu_count': len(cpus),
    'cpu_index': cpus,
    'gpu_count': len(gpus),
    'gpu_index': [gpu['index'] for gpu in gpus],
    'gpu_model': gpus[0]['model'] if gpus else None,
    'threads': process.num_threads(),
    'creation': datetime.datetime.fromtimestamp(process.create_time()).isoformat(),
    'cwd': process.cwd(),
  }


# ==================================================================================================
# Conversion
