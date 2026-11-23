import typing as T

import gymnasium.wrappers
import numpy as np

from tonic import environments


def Gym(
  name: str,
  time_limit: int | T.Literal['default'] | None = 'default',
  time_feature: bool = False,
  scale_actions: bool = True,
  **kwargs,
) -> gymnasium.Env:
  """Returns a wrapped Gym environment."""

  # Build the environment.
  env = gymnasium.make(name, **kwargs)

  # Add ActionRescaler wrapper to normalize action space.
  if scale_actions:
    env = environments.wrappers.ActionRescaler(env)

  # Calculate time limit.
  if time_limit == 'default':
    time_limit = T.cast(int | None, getattr(env, '_max_episode_steps', None))
    if time_limit == float('inf'):
      time_limit = None

  # Add TimeFeature wrapper.
  if time_feature:
    assert time_limit is not None
    env = environments.wrappers.TimeFeature(env, time_limit)

  # Add TimeLimit wrapper.
  if time_limit:
    env = gymnasium.wrappers.TimeLimit(env, time_limit)

  return env


def PyBullet(
  name: str,
  time_limit: int | T.Literal['default'] | None = 'default',
  time_feature: bool = False,
  scale_actions: bool = True,
  **kwargs,
) -> gymnasium.Env:
  """Returns a wrapped PyBullet environment."""

  # Build the environment.
  import pybullet_envs  # type: ignore
  env = gymnasium.make(name, **kwargs)

  # Add ActionRescaler wrapper to normalize action space.
  if scale_actions:
    env = environments.wrappers.ActionRescaler(env)

  # Calculate time limit.
  if time_limit == 'default':
    time_limit = T.cast(int | None, getattr(env, '_max_episode_steps', None))
    if time_limit == float('inf'):
      time_limit = None

  # Add TimeFeature wrapper.
  if time_feature:
    assert time_limit is not None
    env = environments.wrappers.TimeFeature(env, time_limit)

  # Add TimeLimit wrapper.
  if time_limit:
    env = gymnasium.wrappers.TimeLimit(env, time_limit)

  return env


def DMControlSuite(
  domain_name: str,
  task_name: str,
  task_kwargs: T.Optional[dict] = None,
  visualize_reward: bool = True,
  environment_kwargs: T.Optional[dict] = None,
  flatten: bool = True,
  wrappers: list = [],
  time_limit: int | T.Literal['default'] | None = 'default',
  time_feature: bool = False,
  scale_actions: bool = True,
) -> gymnasium.Env:
  '''Returns a wrapped Control Suite environment.'''

  # Build the environment.
  from dm_control import suite  # type: ignore
  from . import adapters
  dmenv = suite.load(
    domain_name=domain_name,
    task_name=task_name,
    task_kwargs=task_kwargs,
    visualize_reward=visualize_reward,
    environment_kwargs=environment_kwargs,
  )
  env = adapters.DMEnv(dmenv, dmenv.task.random)

  if flatten:
    env = environments.wrappers.FlattenObservationAction(env)

  for wrapper in wrappers:
    env = wrapper(env)

  # Add ActionRescaler wrapper to normalize action space.
  if scale_actions:
    env = environments.wrappers.ActionRescaler(env)

  # Calculate time limit.
  if time_limit == 'default':
    time_limit = T.cast(int | None, getattr(dmenv, '_step_limit', None))
    if time_limit == float('inf'):
      time_limit = None

  # Add TimeFeature wrapper.
  if time_feature:
    assert time_limit is not None
    env = environments.wrappers.TimeFeature(env, time_limit)

  # Add TimeLimit wrapper.
  if time_limit:
    env = gymnasium.wrappers.TimeLimit(env, time_limit)

  return env


def DMComposer(
  dmenv,
  flatten: bool = True,
  wrappers: list = [],
  time_limit: int | T.Literal['default'] | None = 'default',
  time_feature: bool = False,
  scale_actions: bool = True,
) -> gymnasium.Env:
  '''Returns a wrapped Composer environment.'''

  # Build the environment.
  from . import adapters
  env = adapters.DMEnv(dmenv, dmenv.random_state)

  if flatten:
    env = environments.wrappers.FlattenObservationAction(env)

  for wrapper in wrappers:
    env = wrapper(env)

  # Add ActionRescaler wrapper to normalize action space.
  if scale_actions:
    env = environments.wrappers.ActionRescaler(env)

  # Calculate time limit.
  if time_limit == 'default':
    time_limit_seconds = getattr(dmenv, '_time_limit', None)
    if time_limit_seconds is not None and time_limit_seconds < float('inf'):
      time_limit = int(np.ceil(time_limit_seconds / dmenv.task.control_timestep))
    else:
      time_limit = None

  # Add TimeFeature wrapper.
  if time_feature:
    assert time_limit is not None
    env = environments.wrappers.TimeFeature(env, time_limit)

  # Add TimeLimit wrapper.
  if time_limit:
    env = gymnasium.wrappers.TimeLimit(env, time_limit)

  return env
