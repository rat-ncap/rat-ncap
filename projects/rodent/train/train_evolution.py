"""
This script trains a policy with an evolution strategy (`evolution2`) on an `MjxEnvObjective`, e.g.:

  python train/train_evolution.py --config-name=1_train/1_4_flxext
"""
import dataclasses
import pathlib
import signal

import hydra
import omegaconf as oc
import torch

import evolution2 as evolution
import infra.launcher  # pylint: disable=unused-import
import mjx_objective as mjxo
import utils.compute
import utils.logging


@dataclasses.dataclass
class TrainEvolutionConfig:
  """Config of an evolution strategy training run.

  Attributes:
    algorithm: Config of the `evolution2.Algorithm`.
    objective: Config of the `mjx_objective.MjxEnvObjective`, without its `logger`. Its
      `train_env_config`, `test_env_config` and `agent_config` stay configs: the objective
      instantiates them.
    trainer: Config of the `evolution2.Trainer`, without its `algorithm`, `objective` and `logger`.
    output_dir: Output directory of the run.
  """
  algorithm: oc.DictConfig
  objective: oc.DictConfig
  trainer: oc.DictConfig
  output_dir: str


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: TrainEvolutionConfig) -> None:
  # Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
  # See: https://github.com/facebookincubator/submitit/issues/1677
  signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))

  # Setup compute and output directory.
  output_dir = pathlib.Path(cfg.output_dir)
  logger = utils.logging.Logger(output_dir)
  logger.info(f'Initialized compute with resources: {utils.compute.jobinfo()}')
  logger.info(f'Initialized output directory: {output_dir}')
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.set_default_device(device)
  logger.info(f'Using device: {device}')

  # Start trainer.
  algorithm: evolution.Algorithm = hydra.utils.instantiate(cfg.algorithm)
  logger.info(f'Instantiated algorithm: {algorithm}')
  objective: mjxo.MjxEnvObjective = hydra.utils.instantiate(cfg.objective, logger=logger)
  logger.info(f'Instantiated objective: {objective}')
  trainer: evolution.Trainer = hydra.utils.instantiate(
    cfg.trainer,
    algorithm,
    objective,
    logger=logger,
    _recursive_=False,
  )

  trainer.initialize()
  trainer.run()


if __name__ == '__main__':
  main()
