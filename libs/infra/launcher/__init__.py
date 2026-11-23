"""
Hydra launch infrastructure, set up by importing this module (e.g. `import infra.launcher` in an
entry point): it registers the `infra` config search path (`/infra/base`, `/compute/*`), the
callbacks that `compute` configs reference, the OmegaConf resolvers that `/infra/base` uses, and a
SIGTERM handler.
"""
import pathlib
import re
import signal

import omegaconf as oc
from hydra.core import config_search_path, plugins
from hydra.experimental import callback
from hydra.plugins import search_path_plugin

import utils.string

# ==================================================================================================
# Configs

# Absolute path to infrastructure configs directory.
CONFIGS_PATH = (pathlib.Path(__file__).parent / 'configs').resolve()


# Extend Hydra config search path to include infrastructure configs. See
# https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin.
class InfraSearchPathPlugin(search_path_plugin.SearchPathPlugin):
  """Search path plugin that adds the infrastructure configs."""
  def manipulate_search_path(self, search_path: config_search_path.ConfigSearchPath) -> None:
    """Appends `CONFIGS_PATH` to the search path."""
    search_path.append(provider='infra', path=f'file://{CONFIGS_PATH}')


plugins.Plugins.instance().register(InfraSearchPathPlugin)

# ==================================================================================================
# Logs


class SubmititLogsCallback(callback.Callback):
  """Callback to symlink Submitit logs to the Hydra job directories."""
  def on_job_start(self, config: oc.DictConfig, **kwargs) -> None:
    """Links the job's Submitit stdout and stderr logs as `stdout.log` and `stderr.log` in its job
    directory, if they exist."""
    job_dir = pathlib.Path(config.hydra.runtime.output_dir)
    job_id = str(config.hydra.job.id)
    submitit_dir = pathlib.Path('../.submitit')  # Use relative path to enable moving sweep dir.
    out_path = submitit_dir / f'{job_id}/{job_id}_0_log.out'
    err_path = submitit_dir / f'{job_id}/{job_id}_0_log.err'
    if (job_dir / out_path).is_file(): (job_dir / 'stdout.log').symlink_to(out_path)
    if (job_dir / err_path).is_file(): (job_dir / 'stderr.log').symlink_to(err_path)


# ==================================================================================================
# Resolvers

# Register resolver to convert `override_dirname` to shorter and path-safe directory name, without
# slashes and only including the retargeted option key/value (e.g. "/group1/group2@foo=option1" ->
# "foo=option1").
_OVERRIDES_PATTERN = re.compile(r"([a-zA-Z0-9._-]+@)?([a-zA-Z0-9._-]+=)")
oc.OmegaConf.register_new_resolver(
  "overrides",
  lambda s: _OVERRIDES_PATTERN.sub(r"\2", s.replace('/', '.')),
)

# Register resolver to create a human-readable unique ID.
oc.OmegaConf.register_new_resolver('hruid', utils.string.hruid, use_cache=True)


# Register resolver to create an experiment ID from the given name or config name.
def _exp_id(name: str | None, config_name: str) -> str:
  """Returns the experiment ID: `name`, or else the config name, without a YAML extension."""
  exp_id = name or config_name
  exp_id = exp_id.removesuffix('.yaml').removesuffix('.yml')
  return exp_id


oc.OmegaConf.register_new_resolver('exp_id', _exp_id, use_cache=True)


# Register resolver to create a sweep ID from the timestamp, hruid, and tag.
def _sweep_id(now: str, hruid: str, tag: str | None) -> str:
  """Returns the sweep ID `<now>-[<tag>:]<hruid>`."""
  sweep_id = f"{now}-{tag + ':' if tag else ''}{hruid}"
  return sweep_id


oc.OmegaConf.register_new_resolver('sweep_id', _sweep_id, use_cache=True)

# ==================================================================================================
# Signals

# Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
# This code will need to be run within the Hydra entrypoint function as well.
# See: https://github.com/facebookincubator/submitit/issues/1677
signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))
