"""
This script rolls out a policy in freshly initialized environments, logs the returns of each
rollout to `test.csv`, and saves each rollout's data.
"""
import dataclasses
import pathlib
import signal
import time
import typing as T

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf as oc
import torch

import composer as cx
import composer.utils.rollout as cxr
import infra.launcher  # pylint: disable=unused-import
import utils.compute
import utils.io
import utils.logging


@dataclasses.dataclass
class TestConfig:
  """Config of a test.

  Attributes:
    output_dir: Output directory of the test.
    env: Config of the environment.
    subtasks: Subtask values, {name: [value of each subtask]}. Each of the `population` policy
      copies is rolled out in every subtask: world `p * subtasks + s` is copy `p` in subtask `s`.
    rollout_kwargs: Arguments of `cxr.rollout_from_policy` (e.g. its `policy` and `record`), but
      not `vectorize`, `seed` or `subtask`, which the test sets.
    population: Number of copies of the policy.
    rollouts: Number of rollouts, each in newly initialized environments.
    seed: Base seed, from which each rollout's seed follows.
  """
  output_dir: str

  env: oc.DictConfig
  subtasks: T.Mapping[str, list[int | float]]
  rollout_kwargs: cxr.RolloutFromPolicyKwargs

  population: int = 1
  rollouts: int = 1
  seed: int = 0


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: TestConfig) -> None:
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

  # Set up and validate rollout kwargs.
  rollout_kwargs = setup_rollout_kwargs(cfg, logger)
  population = cfg.population
  vectorize = rollout_kwargs['vectorize']

  start_time = time.time()
  for rollout, seed in enumerate(get_rollout_seeds(cfg.seed, cfg.rollouts)):
    test_time = time.time()
    rollout_kwargs['seed'] = seed

    # Re-initialize (and re-jit) the environment every rollout so that randomized arenas (e.g.
    # bumps) are regenerated.
    env_tuple = initialize_env(cfg, seed, vectorize)
    with torch.no_grad():
      data = cxr.rollout_from_policy(
        env_tuple,
        **rollout_kwargs,
      )

    # Compute and log statistics: the mean reward per step, from the rewards of the T steps after
    # the reset, as in training.
    reward = jnp.stack(data.reward[1:])  # [time, vectorize]
    reward = reward.reshape(-1, population, vectorize // population)
    subtask_returns = reward.mean(0)  # [population, subtasks]
    returns = subtask_returns.mean(-1)  # [population]

    curr_time = time.time()
    checkpoint = getattr(rollout_kwargs['policy'], 'checkpoint', None)
    logger.table('test.csv').store({
      'test/rollout': rollout,
      'test/checkpoint_epoch': getattr(checkpoint, 'epoch', -1),
      'test/seed': seed,
      'test/seconds': curr_time - start_time,
      'test/rollout_seconds': curr_time - test_time,
    }).store({
      'test/return': np.array(returns).mean().item(),
      **{
        f'test/subtask_return_{i}': np.array(v).mean().item()
        for i, v in enumerate(subtask_returns.T)
      },
    }).save()

    # Save rollout data.
    rollout_dir = output_dir / f'rollouts/{rollout}'
    rollout_dir.mkdir(parents=True, exist_ok=True)
    save_rollout(data, rollout_dir)
    logger.info(f'Completed rollout {rollout}.')


def save_rollout(data: cxr.Rollout, rollout_dir: pathlib.Path) -> None:
  """Saves a rollout to `rollout_dir`: its records, rewards and rates as `rollout.pkl` (a dict of
  `record`, `record_rate`, `reward` and `render_rate`), and its frames, if rendered, as
  `render.npy`."""
  if data.render:
    np.save(rollout_dir / 'render.npy', np.array(data.render))
  utils.io.write_file(rollout_dir / 'rollout.pkl', dict(
    render_rate=data.render_rate,
    record=data.record,
    record_rate=data.record_rate,
    reward=data.reward,
  ))


def get_rollout_seeds(seed: int, rollouts: int) -> list[int]:
  """Returns the seed of each rollout, split in turn from the base `seed`'s key so that tests of
  different base seeds don't share rollout seeds."""
  rng = jax.random.key(seed)
  seeds = []
  for _ in range(rollouts):
    rng, rng_rollout = jax.random.split(rng)
    seeds.append(int(jax.random.randint(rng_rollout, (), 0, jnp.iinfo(jnp.int32).max)))
  return seeds


def setup_rollout_kwargs(
  cfg: TestConfig,
  logger: utils.logging.Logger,
) -> cxr.RolloutFromPolicyKwargs:
  """Returns the instantiated rollout kwargs, with `vectorize` = population * subtasks, the
  subtask of each world, and defaults set."""
  population = cfg.population
  subtasks = T.cast(dict[str, list[int | float]], oc.OmegaConf.to_container(cfg.subtasks))
  rollouts = cfg.rollouts
  # Instantiate the policy and record functions.
  rollout_kwargs: cxr.RolloutFromPolicyKwargs = hydra.utils.instantiate(
    cfg.rollout_kwargs,
    _convert_='object',
  )

  # Validate subtasks.
  subtask_count = len(next(iter(subtasks.values()))) if subtasks else 1
  if subtasks and any(len(v) != subtask_count for v in subtasks.values()):
    raise ValueError(f'All subtasks must have the same length: {subtasks}')

  logger.info(
    f'Starting {rollouts} rollouts of population size {population} and {subtask_count} subtasks.'
  )
  logger.info(f'vectorize = population * subtasks = {population * subtask_count}')

  # Set default rollout kwargs.
  assert 'vectorize' not in rollout_kwargs, '`vectorize` is calculated from population and subtasks'
  assert 'seed' not in rollout_kwargs, '`seed` is evolved per rollout from the top-level `seed`'
  assert 'subtask' not in rollout_kwargs, 'Set `subtasks` in the config instead'
  rollout_kwargs['vectorize'] = population * subtask_count
  rollout_kwargs['subtask'] = cx.tile_value(
    {k: jnp.array(v) for k, v in subtasks.items()},
    population,
  )
  rollout_kwargs['record_rate'] = rollout_kwargs.get('record_rate', 1 / cfg.env.task.env_timestep)
  rollout_kwargs['record'] = rollout_kwargs.get('record', {})

  return rollout_kwargs


def initialize_env(
  cfg: TestConfig,
  seed: int,
  vectorize: int,
) -> tuple[cx.MjxEnv, cxr.EnvResetFn, cxr.EnvStepFn, cx.MjxEnvState]:
  """Returns the instantiated environment of `vectorize` worlds, its jitted reset and step, and its
  initial state."""
  env: cx.MjxEnv = hydra.utils.instantiate(cfg.env, _convert_='object')
  rng = jax.random.split(jax.random.key(seed), vectorize)  # [vectorize]
  state = env.initialize(rng)
  return (
    env,                        # env
    eqx.filter_jit(env.reset),  # reset
    eqx.filter_jit(env.step),   # step
    state,                      # state
  )


if __name__ == '__main__':
  main()
