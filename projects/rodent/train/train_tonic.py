"""
This script trains a policy with a Tonic reinforcement learning agent (e.g. PPO) in vectorized
Composer environments, e.g.:

  python train/train_tonic.py --config-name=1_train/4_3_mlp_ppo_flxext
"""
import dataclasses
import pathlib
import signal
import typing as T

import hydra
import jax
import omegaconf as oc
import torch

import composer as cx
import composer.utils.gym as cxg
import infra.launcher  # pylint: disable=unused-import
import mjx_objective as mjxo
import tonic
import tonic.logging
import tonic.torch.agent
import utils.compute

# Tonic's logger, configured in `main` (`tonic.logging.basicConfig`).
logger = tonic.logging.getLogger()


@dataclasses.dataclass
class TrainTonicConfig:
  """Config of a Tonic training run.

  Attributes:
    agent: Config of the `tonic.torch.agent.Agent`.
    train_env: Config of the training `composer.MjxEnv` (with autoreset).
    test_env: Config of the test `composer.MjxEnv` (without autoreset).
    train_size: Number of training environments per subtask.
    test_size: Number of test environments per subtask.
    train_subtasks: Subtask values of the training environments, {name: [value of each subtask]},
      or None.
    test_subtasks: Subtask values of the test environments, as `train_subtasks`.
    trainer: Config of the `tonic.Trainer`.
    seed: Seed of the run.
    output_dir: Output directory of the run.
  """
  agent: oc.DictConfig
  train_env: oc.DictConfig
  test_env: oc.DictConfig
  train_size: int
  test_size: int
  train_subtasks: dict[str, list[int] | list[float] | list[bool]] | None
  test_subtasks: dict[str, list[int] | list[float] | list[bool]] | None
  trainer: oc.DictConfig
  seed: int
  output_dir: str


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: TrainTonicConfig) -> None:
  # Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
  # See: https://github.com/facebookincubator/submitit/issues/1677
  signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))

  # Setup compute and output directory.
  output_dir = pathlib.Path(cfg.output_dir)
  tonic.logging.basicConfig(log_dir=output_dir, level='info')
  logger.info(f'Initialized compute with resources: {utils.compute.jobinfo()}')
  logger.info(f'Initialized output directory: {output_dir}')
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.set_default_device(device)
  logger.info(f'Using device: {device}')

  # Instantiate agent and environments.
  agent: tonic.torch.agent.Agent = hydra.utils.instantiate(cfg.agent, _convert_='object')
  train_env_composer: cx.MjxEnv = hydra.utils.instantiate(cfg.train_env, _convert_='object')
  test_env_composer: cx.MjxEnv = hydra.utils.instantiate(cfg.test_env, _convert_='object')
  train_env = T.cast(tonic.VectorEnv, cxg.ComposerEnv(train_env_composer))
  test_env = T.cast(tonic.VectorEnv, cxg.ComposerEnv(
    test_env_composer,
    autoreset=False,
  ))
  train_subtasks = None if cfg.train_subtasks is None else {
    k: torch.tensor(v) for k, v in cfg.train_subtasks.items()
  }
  test_subtasks = None if cfg.test_subtasks is None else {
    k: torch.tensor(v) for k, v in cfg.test_subtasks.items()
  }

  # Start trainer.
  trainer: tonic.Trainer = hydra.utils.instantiate(cfg.trainer, _convert_='object')
  trainer.initialize(
    agent=agent,
    train_environment=train_env,
    train_batch_size=cfg.train_size,
    train_subtasks=train_subtasks,
    test_environment=test_env,
    test_batch_size=cfg.test_size,
    test_subtasks=test_subtasks,
  )

  # Save the configs and spaces next to the trainer's `checkpoint/{epoch}.pt` models, in the
  # checkpoint layout of `mjx_objective` (see `mjx_objective.load_checkpoint`).
  state = test_env_composer.initialize(jax.random.split(jax.random.key(cfg.seed), 1))
  mjxo.save_checkpoint_spec(
    output_dir / 'checkpoint',
    env_config=oc.OmegaConf.create(oc.OmegaConf.to_container(cfg.test_env, resolve=True)),
    agent_config=oc.OmegaConf.create(oc.OmegaConf.to_container(cfg.agent, resolve=True)),
    subtask_space=test_env_composer.subtask_space(state).unvectorize(),
    observation_space=test_env_composer.observation_space(state).unvectorize(),
    action_space=test_env_composer.action_space(state).unvectorize(),
  )
  trainer.run()


if __name__ == '__main__':
  main()
