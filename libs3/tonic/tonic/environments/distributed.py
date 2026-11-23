import multiprocessing
import multiprocessing.connection
import typing as T

import torch

from tonic import core


class Sequential(core.VectorEnv):
  """A group of environments to step in sequence."""
  # TODO: Make this work with dict observations and actions.

  def __init__(
    self,
    environment_builder: T.Callable[[], core.Env],
    sequential: int,
  ):
    self.environments = [environment_builder() for _ in range(sequential)]
    self.observation_space = self.environments[0].observation_space
    self.action_space = self.environments[0].action_space
    self.num_envs = sequential

  def initialize(self, seed: int):
    for i, env in enumerate(self.environments):
      env.initialize(seed + i)

  def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[core.Observation, dict[str, T.Any]]:
    observations_list: list[core.Observation] = []
    infos = []
    for i, environment in enumerate(self.environments):
      env_seed = seed + i if seed is not None else None
      observation, info = environment.reset(seed=env_seed)
      observations_list.append(observation)
      infos.append(info)
    observations = core.stack_tensors(observations_list)
    return observations, {}

  def step(self, actions: core.Action) -> tuple[core.Observation, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, T.Any]]:
    rewards = []
    terminations = []
    truncations = []
    next_observations = []  # Observations for transition (final if terminated/truncated).
    new_observations = []  # Observations for next step (after reset if terminated/truncated).

    for i in range(len(self.environments)):
      action = {k: v[i] for k, v in actions.items()} if isinstance(actions, dict) else actions[i]
      next_observation, reward, terminated, truncated, info = self.environments[i].step(action)
      rewards.append(reward)
      terminations.append(terminated)
      truncations.append(truncated)
      next_observations.append(next_observation)
      if terminated or truncated:
        # This "same-step" auto-reset behavior simplifies usage: each call to `step()` returns a 
        # valid transition that does not cross episode boundaries. In contrast, using "next-step" 
        # auto-reset behavior would require discarding transitions that cross episode boundaries and
        # managing variable-length transition batches.
        new_observation, _ = self.environments[i].reset()
      else:
        new_observation = next_observation
      new_observations.append(new_observation)

    next_observations = torch.stack(next_observations)
    new_observations = torch.stack(new_observations)
    rewards = torch.tensor(rewards)
    terminations = torch.tensor(terminations, dtype=torch.bool)
    truncations = torch.tensor(truncations, dtype=torch.bool)

    # TODO: Collate and return info dicts from env steps.
    infos = {'new_observations': new_observations}
    
    return next_observations, rewards, terminations, truncations, infos
  
  def close(self) -> None:
    """Close all environments."""
    for env in self.environments:
      env.close()

  def render(self) -> None:
    """Render the environments."""
    raise NotImplementedError()


class Parallel(core.VectorEnv):
  """A group of environments to step in parallel."""
  # TODO: Make this work with dict observations and actions.

  def __init__(
    self,
    environment_builder: T.Callable[[], core.Env],
    parallel: int,
    sequential: int,
  ):
    self.environment_builder = environment_builder
    self.parallel = parallel
    self.sequential = sequential
    self.num_envs = parallel * sequential

  def initialize(self, seed: int) -> None:
    """Initialize worker processes."""
    def worker(action_pipe: multiprocessing.connection.Connection, index: int, seed: int) -> None:
      """Process holding a sequential group of environments."""
      envs = Sequential(self.environment_builder, self.sequential)
      envs.initialize(seed)
      obs, _ = envs.reset(seed=seed)

      self.output_queue.put((index, obs, {}))
      while True:
        actions = action_pipe.recv()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        step_dict = {
          'observations': obs,
          'rewards': rewards,
          'terminations': terminations,
          'truncations': truncations,
          'infos': infos
        }
        self.output_queue.put((index, infos['new_observations'], step_dict))

    # Store observation and action spaces.
    dummy_environment = self.environment_builder()
    self.observation_space = dummy_environment.observation_space
    self.action_space = dummy_environment.action_space
    del dummy_environment
    self.started = False

    # Start worker processes.
    self.output_queue = multiprocessing.Queue[tuple[int, core.Observation, dict[str, T.Any]]]()
    self.action_pipes: list[multiprocessing.connection.Connection] = []
    for i in range(self.parallel):
      pipe, worker_end = multiprocessing.Pipe()
      self.action_pipes.append(pipe)
      group_seed = seed + i * self.sequential
      process = multiprocessing.Process(target=worker, args=(worker_end, i, group_seed))
      process.daemon = True
      process.start()

  def reset(
    self,
    seed: int | None = None,
    options: dict | None = None
  ) -> tuple[torch.Tensor, dict[str, T.Any]]:
    assert not self.started
    self.started = True
    new_observations_list: list[core.Observation] = [torch.empty()] * self.parallel
    infos_list: list[dict[str, T.Any]] = [{}] * self.parallel
    for _ in range(self.parallel):
      index, new_observations, info = self.output_queue.get()
      new_observations_list[index] = new_observations
      infos_list[index] = info

    self.new_observations_list = torch.stack(new_observations_list)
    self.observations_list = torch.zeros_like(self.new_observations_list)
    self.rewards_list = torch.zeros((self.parallel, self.sequential))
    self.terminations_list = torch.zeros((self.parallel, self.sequential), dtype=torch.bool)
    self.truncations_list = torch.zeros((self.parallel, self.sequential), dtype=torch.bool)

    # TODO: We should return an infos dict, but infos_list needs to be consolidated from
    #       list[dict[str, T.Any]] -> dict[str, T.Any].
    return torch.concatenate(new_observations_list), {}

  def step(self, actions: core.Action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, T.Any]]:
    actions_list = torch.split(actions, self.parallel)
    for actions, pipe in zip(actions_list, self.action_pipes):
      pipe.send(actions)

    for _ in range(self.parallel):
      index, new_observations, step_dict = self.output_queue.get()
      self.new_observations_list[index] = new_observations
      if step_dict:  # Skip on reset
        self.observations_list[index] = step_dict['observation']
        self.rewards_list[index] = step_dict['reward']
        self.terminations_list[index] = step_dict['termination']
        self.truncations_list[index] = step_dict['truncation']

    new_observations_parallel = self.new_observations_list.flatten(0, 1)
    observations_parallel = self.observations_list.flatten(0, 1)
    rewards_parallel = self.rewards_list.flatten(0, 1)
    terminations_parallel = self.terminations_list.flatten(0, 1)
    truncations_parallel = self.truncations_list.flatten(0, 1)
    info = {
      'new_observations': new_observations_parallel
    }

    return observations_parallel, rewards_parallel, terminations_parallel, truncations_parallel, info
  
  def close(self) -> None:
    """Close all worker processes."""
    for pipe in self.action_pipes:
      pipe.close()

  def render(self) -> None:
    """Render the environments."""
    raise NotImplementedError()


def distribute(
  environment_builder: T.Callable[[], core.Env],
  parallel: int = 1,
  sequential: int = 1,
) -> T.Union[Sequential, Parallel]:
  """Distributes workers over parallel and sequential groups."""
  if parallel <= 1:
    return Sequential(environment_builder, sequential=sequential)
  else:
    return Parallel(environment_builder, parallel=parallel, sequential=sequential)
