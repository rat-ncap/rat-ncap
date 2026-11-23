import pathlib

import hydra
import omegaconf as oc
import pytest

import infra.launcher

CONFIGS_PATH = pathlib.Path(infra.launcher.__file__).parent / 'configs'


def compose(*overrides: str) -> oc.DictConfig:
  """Returns `/infra/base` composed with the overrides, with the Hydra config."""
  with hydra.initialize_config_dir(str(CONFIGS_PATH), version_base=None):
    return hydra.compose('infra/base', overrides=list(overrides), return_hydra_config=True)


def test_base_defaults_to_local_compute():
  config = compose()

  assert config.hydra.launcher._target_.endswith('LocalLauncher')
  assert config.hydra.mode == hydra.types.RunMode.MULTIRUN
  assert config.compute.gpu == 0


@pytest.mark.parametrize(
  'override_dirname, expected',
  [
    ('seed=1', 'seed=1'),
    ('/body/rat/controller=pos,seed=2', '.body.rat.controller=pos,seed=2'),
    ('model@agent.model=mlp', 'agent.model=mlp'),
  ],
  ids=['plain', 'group_path', 'retargeted_package'],
)
def test_overrides_resolver_makes_a_path_safe_directory_name(override_dirname, expected):
  config = oc.OmegaConf.create(dict(dirname=override_dirname, name='${overrides:${dirname}}'))

  assert config.name == expected
  assert '/' not in config.name


def test_exp_id_defaults_to_the_config_name_without_its_suffix():
  config = oc.OmegaConf.create(dict(a='${exp_id:null,1_train/1_4_flxext.yaml}', b='${exp_id:x,y}'))

  assert (config.a, config.b) == ('1_train/1_4_flxext', 'x')
