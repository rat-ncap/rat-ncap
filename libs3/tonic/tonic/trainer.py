import time
import typing as T

import torch

from tonic import core, logging
from tonic.torch import agent


class Trainer:
  """Trainer used to train and evaluate an agent on an environment."""

  def __init__(
    self,
    seed: int = 0,
    epochs: int = 500,
    epoch_steps: int = 40,
    test_freq: int = 10,
    save_freq: int | None = None,
  ):
    self.seed = seed
    self.epochs = epochs
    self.epoch_steps = epoch_steps
    self.test_freq = test_freq
    self.save_freq = test_freq if save_freq is None else save_freq
    self.logger = logging.getLogger()

  def initialize(
    self,
    agent: agent.Agent,
    train_environment: core.VectorEnv,
    train_batch_size: int,
    train_subtasks: dict[str, torch.Tensor] | None = None,
    test_environment: core.VectorEnv | None = None,
    test_batch_size: int | None = None,
    test_subtasks: dict[str, torch.Tensor] | None = None,
  ) -> None:
    """
    Initializes the agent and train/test environments with the given random seed. Expects the 
    train environment to perform same-step autoreset (so that the same number of steps are recorded 
    every epoch). Expects the test environment to be episodic (as rollout will continue until all
    parallel environments have terminated/truncated). The agent is initialized with the observation 
    and action spaces of the train environment, which is expected to be the same as the test 
    environment. The number of subtasks defaults to 1 if None or an empty dictionary is provided.

    Args:
      agent: Agent to be trained and evaluated.
      train_environment: Environment used for training.
      train_batch_size: Number of parallel environments used per subtask for training.
      train_subtasks: A mapping of task parameter key names to [subtask] values for that parameter.
      test_environment: Environment used for evaluation.
      test_batch_size: Number of parallel environments used per subtask for evaluation.
      test_subtasks: A mapping of task parameter key names to [subtask] values for that parameter.
    """
    self.agent = agent
    self.train_environment = train_environment
    self.test_environment = test_environment
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size or 1
    
    self.train_subtask_count = len(next(iter(train_subtasks.values()))) if train_subtasks else 1
    self.test_subtask_count = len(next(iter(test_subtasks.values()))) if test_subtasks else 1
    assert self.train_batch_size % self.train_subtask_count == 0, (
      f'`train_batch_size` ({self.train_batch_size}) must be divisible by '
      f'number of subtasks ({self.train_subtask_count})'
    )
    assert self.test_batch_size % self.test_subtask_count == 0, (
      f'`test_batch_size` ({self.test_batch_size}) must be divisible by '
      f'number of subtasks ({self.test_subtask_count})'
    )
    self.train_subtasks = core.tile_tensors(
      train_subtasks or {},
      self.train_batch_size // self.train_subtask_count,
    )  # dict[str, torch.Tensor] with shape [`train_batch_size`]
    self.test_subtasks = core.tile_tensors(
      test_subtasks or {},
      self.test_batch_size // self.test_subtask_count,
    )  # dict[str, torch.Tensor] with shape [`test_batch_size`]

    generator = torch.Generator(device=torch.get_default_device()).manual_seed(self.seed)
    dtype_info = torch.iinfo(torch.int32)
    
    rng = torch.randint(
      dtype_info.min,
      dtype_info.max,
      (train_batch_size,),
      generator=generator,
    )
    self.train_environment.initialize(rng)

    if self.test_environment:
      assert test_batch_size is not None
      rng = torch.randint(
        dtype_info.min,
        dtype_info.max, 
        (test_batch_size,),
        generator=generator,
      )
      self.test_environment.initialize(rng)

    observation_space = self.train_environment.observation_space
    action_space = self.train_environment.action_space
    assert isinstance(observation_space, core.ObservationSpace)
    assert isinstance(action_space, core.ActionSpace)
    self.agent.initialize(observation_space, action_space, self.seed)

  def run(self) -> None:
    """Runs the main training loop."""
    self.logger.info(f'Started run for {self.epochs } epochs')
    start_time = last_epoch_time = time.time()

    # Start the environments. Expected to perform same-step autoreset automatically.
    observations, _ = self.train_environment.reset(
      seed=self.seed,
      options=dict(subtask=self.train_subtasks),
    )

    for epoch in range(self.epochs):
      # Evaluate the agent on the test environment.
      if epoch % self.test_freq == 0 and self.test_environment:
        self._test(epoch, start_time)

      # Save the current params and a checkpoint at the start of the epoch.
      if epoch % self.save_freq == 0:
        self._save(epoch)

      # Run the epoch.
      average_rewards = torch.zeros(self.train_batch_size)
      resets = torch.zeros(self.train_batch_size, dtype=torch.long)
      for _ in range(self.epoch_steps):
        # Select actions and step the environment.
        actions = self.agent.step(observations)
        next_observations, rewards, terminations, truncations, infos = (
          self.train_environment.step(actions)
        )
        self.agent.record(
          observations=observations,
          actions=actions,
          rewards=rewards,
          terminations=terminations,
          truncations=truncations,
          next_observations=next_observations,
        )
        observations = infos['new_observations']

        # Update training statistics.
        average_rewards += rewards / self.epoch_steps
        resets += (terminations | truncations).long()

      # Update agent.
      agent_metrics = self.agent.update() or {}
      
      # Log the data.
      current_time = time.time()
      epoch_time = current_time - last_epoch_time
      self.logger.table('train.csv').store({
        'train/epochs': epoch,
        'train/seconds': current_time - start_time,
        'train/epoch_seconds': epoch_time,
        'train/resets': resets.numpy(force=True),
        'train/returns/normalized': average_rewards.numpy(force=True),
      }).store({
        f'train/rewards/subtask{subtask}': rewards.mean().item() for subtask, rewards in enumerate(
          average_rewards.numpy(force=True).reshape((-1, self.train_subtask_count)).T,
        )
      }).store({
        f'train/{k}': v if isinstance(v, (int, float)) else v.numpy(force=True)
        for k, v in agent_metrics.items()
      }).save()
      self.logger.info(f'Epoch {epoch}: Completed train step')
      last_epoch_time = time.time()
    
    # Save the final params and checkpoint.
    self._save(epoch + 1)

  def _save(self, epoch: int) -> None:
    actor = T.cast(torch.nn.Module, self.agent.model.actor)
    actor_params: dict[str, torch.Tensor | dict] = dict(actor.state_dict())
    params = core.pack_tensors(dict(core.flatten_tensors(actor_params)))
    self.logger.file('params').save(f'{epoch}.pkl', params.cpu())
    self.agent.save(str(self.logger.log_dir / f'checkpoint/{epoch}'))
      
  def _test(self, epochs: int, start_time: float) -> None:
    """Tests the agent on the test environment."""
    assert self.test_environment and self.test_batch_size
    test_time = time.time()

    # Test loop.
    returns = torch.zeros(self.test_batch_size)
    lengths = torch.zeros(self.test_batch_size, dtype=torch.long)

    observations, _ = self.test_environment.reset(
      seed=self.seed,
      options=dict(subtask=self.test_subtasks),
    )
    running = torch.ones(self.test_batch_size, dtype=torch.bool)
    while running.any():
      # Select an action and take a step in the environment.
      actions = self.agent.test_step(observations)
      observations, rewards, terminations, truncations, infos = self.test_environment.step(actions)

      # Accumulate metrics for active environments BEFORE marking episodes as done.
      # This ensures the final step (that caused termination) is properly counted.
      returns += rewards * running
      lengths += running.long()
      
      # Update running state for next iteration
      running = running & ~(terminations | truncations)

    # Log the data.
    self.logger.table('test.csv').store({
      'test/epochs': epochs,
      'test/seconds': time.time() - start_time,
      'test/epoch_seconds': time.time() - test_time,
      'test/lengths': lengths.numpy(force=True),
      'test/returns': returns.numpy(force=True),
    }).store({
      f'test/returns/subtask{subtask}': rewards.mean().item() for subtask, rewards in enumerate(
        returns.numpy(force=True).reshape((-1, self.test_subtask_count)).T,
      )
    }).save()
