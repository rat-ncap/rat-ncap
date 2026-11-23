"""
This module provides the core classes that coordinate train/test logic.

The `Objective` abstracts a fitness function that scores a parameter vector (e.g. an episodic
return score by rolling out a neural network with the given parameters).

The `Algorithm` manages the computations of a single generation, which involves "spawning"
children parameter vectors from parent vectors, "scoring" these children using the `Objective`, and
"selecting" the children to become the next generation's parents.

The `Trainer` orchestrates the train/test loop, which involves running the `Algorithm` for many
generations and data logging.
"""
import abc
import dataclasses
import os
import pathlib
import time
import typing as T

import pandas as pd
import torch
import tqdm

import utils.io
import utils.logging as utlog

__all__ = ['Objective', 'AlgorithmTrainData', 'AlgorithmTestData', 'Algorithm', 'Trainer']

# ==================================================================================================
# Objective


class Objective(abc.ABC):
  """An objective function that assigns scores to parameter vectors. The algorithm aims to maximize
  this function by evolving the parameter vectors."""
  def initialize(self, seed: int, train_size: int, test_size: int) -> None:
    """Initializes the objective once before the first generation.

    Args:
      seed: Random seed.
      train_size: Number of parameter vectors scored per train step.
      test_size: Number of parameter vectors scored per test step.
    """
    pass

  @abc.abstractmethod
  def get_initial_params(self, seeds: torch.Tensor) -> torch.Tensor:
    """Returns a batch of initial parameter vectors for the first generation.

    Args:
      seeds: Random seeds, int64[population_size].

    Returns:
      params: Initial parameter vectors, [population_size, param_size].
    """
    raise NotImplementedError()

  def get_updated_params(self, params: torch.Tensor) -> torch.Tensor:
    """Returns a batch of updated parameter vectors for the next generation, given the proposed
    parameter vectors. This is useful for modifying/constraining parameters between generations.

    Args:
      params: Proposed parameter vectors, [population_size, param_size].

    Returns:
      params: Updated parameter vectors, [population_size, param_size].
    """
    # By default, accept all proposed parameters without modification.
    return params

  @abc.abstractmethod
  def get_score(
    self,
    params: torch.Tensor,
    seeds: torch.Tensor,
    *,
    mode: T.Literal['train', 'test'],
    epoch: int = 0,
    stats: dict[str, float | torch.Tensor],
    files: dict[str, T.Any],
  ) -> torch.Tensor:
    """Evaluates this objective on the given parameter vectors.

    Args:
      params: Parameter vectors to evaluate, [size, param_size], where `size` is `train_size` or
        `test_size` (see `initialize`).
      seeds: Random seeds, int64[size].
      mode: Whether to evaluate in `train` or `test` mode.
      epoch: Current epoch number.
      stats: Stats dict that this method adds to, saved as columns of the train or test table.
      files: Files dict that this method adds to, each a pickleable object saved to its own file.

    Returns:
      score: Objective score (i.e. fitness value) to maximize, [size].
    """
    raise NotImplementedError()

  def on_run_start(
    self,
    *,
    algorithm: 'Algorithm',
  ) -> None:
    """Callback executed by the `Trainer` at the start of the run."""
    pass

  def on_run_end(
    self,
    *,
    algorithm: 'Algorithm',
  ) -> None:
    """Callback executed by the `Trainer` before the end of the run."""
    pass

  def on_train_epoch_start(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
  ) -> None:
    """Callback executed by the `Trainer` before the start of a train epoch."""
    pass

  def on_train_epoch_end(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
    train_data: 'AlgorithmTrainData',
  ) -> None:
    """Callback executed by the `Trainer` after the end of a train epoch."""
    pass

  def on_test_epoch_start(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
  ) -> None:
    """Callback executed by the `Trainer` before the start of a test epoch."""
    pass

  def on_test_epoch_end(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
    test_data: 'AlgorithmTestData',
  ) -> None:
    """Callback executed by the `Trainer` after the end of a test epoch."""
    pass

  def on_save(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
    train_data: 'AlgorithmTrainData',
    test_data: 'AlgorithmTestData | None',
    log_dir: pathlib.Path,
  ) -> None:
    """Callback executed by the `Trainer` before saving recorded data."""
    pass


# ==================================================================================================
# Algorithm


@dataclasses.dataclass
class AlgorithmTrainData:
  """Data collected during a train step.

  Attributes:
    params: The new parent parameter vectors, [population_size, param_size].
    best_param: The best-scoring child parameter vector, [param_size].
    stats: Stats, saved as columns of the train table.
    files: Pickleable objects, each saved to its own file.
  """
  params: torch.Tensor
  best_param: torch.Tensor
  stats: dict[str, float | torch.Tensor]
  files: dict[str, T.Any]


@dataclasses.dataclass
class AlgorithmTestData:
  """Data collected during a test step.

  Attributes:
    scores: Test scores, [test_size].
    stats: Stats, saved as columns of the test table.
    files: Pickleable objects, each saved to its own file.
  """
  scores: torch.Tensor
  stats: dict[str, float | torch.Tensor]
  files: dict[str, T.Any]


class Algorithm(abc.ABC):
  """An algorithm's centralized logic for spawning children parameter vectors from parent vectors,
  and selecting new parents from the scored children. The trainer process stores a single instance
  of this class."""
  @abc.abstractmethod
  def initialize(self, objective: Objective, seed: int) -> torch.Tensor:
    """Initializes the algorithm with the given objective, and returns the initial parameter
    vectors.

    Args:
      objective: Objective instance.
      seed: Random seed.

    Returns:
      params: Initial parameter vectors, [population_size, param_size].
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, epoch: int, seed: int) -> AlgorithmTrainData:
    """Executes a single training step consisting of a spawn-score-select cycle.

    Args:
      epoch: Current epoch.
      seed: Random seed.

    Returns:
      Train data (e.g. parameter vectors, statistics, artifacts).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def test(self, epoch: int, seed: int) -> AlgorithmTestData:
    """Scores the parent parameter vectors with the objective in test mode.

    Args:
      epoch: Current epoch.
      seed: Random seed.

    Returns:
      Test data (e.g. scores, statistics, artifacts).
    """
    raise NotImplementedError()

  def get_state(self) -> T.Any:
    """Returns the picklable state the algorithm needs to continue training (e.g. its parameter
    point and optimizer state). The trainer saves it with the parameters to resume runs."""
    raise NotImplementedError(f'{type(self).__name__} cannot save its state to resume.')

  def set_state(self, state: T.Any) -> None:
    """Restores, after `initialize`, a state returned by `get_state` to continue training from
    it."""
    raise NotImplementedError(f'{type(self).__name__} cannot restore a state to resume.')


# ==================================================================================================
# Trainer


class Trainer:
  """The `Trainer` orchestrates the train/test loop and data logging. It maximizes the given
  `Objective` using the given `Algorithm`."""
  def __init__(
    self,
    algorithm: Algorithm,
    objective: Objective,
    epochs: int = 1000,
    seed: int = 0,
    test_freq: int = 1,
    save_freq: int = 5,
    show_progress: bool = False,
    logger: os.PathLike | str | utlog.Logger = './outputs',
    resume: bool | T.Literal['auto'] = 'auto',
    resume_from: os.PathLike | str | None = None,
  ):
    """Initializes the trainer. Call `initialize` before `run`.

    Args:
      algorithm: Algorithm instance.
      objective: Objective instance.
      epochs: Number of train epochs.
      seed: Global random seed.
      test_freq: Test frequency (epochs).
      save_freq: Save frequency (epochs).
      show_progress: Whether to show epoch progress bar.
      logger: Logger instance or log directory.
      resume: Whether to resume from the newest algorithm state saved in this run's log directory
        (`algorithm/{epoch}.pkl`): if there is one ('auto'), always (True, error if there is none),
        or never (False). A Slurm requeue (e.g. preemption) reruns the job in the same log
        directory, so it continues where it stopped.
      resume_from: Another run's log directory to resume from instead (its newest algorithm state,
        and its train/test rows up to that epoch copied here). If given, the run resumes from it
        regardless of `resume`.

    Resuming is exact: each epoch's mutations and environment seeds derive from (seed, epoch), and
    the epochs after the last save are recomputed. This assumes the objective keeps no state across
    epochs, as only the algorithm's state is saved.
    """
    self.algorithm = algorithm
    self.objective = objective
    self.epochs = epochs
    self.seed = seed
    self.test_freq = test_freq
    self.save_freq = save_freq
    self.show_progress = show_progress
    self.logger = logger if isinstance(logger, utlog.Logger) else utlog.Logger(logger)
    self.resume = resume
    self.resume_from = pathlib.Path(resume_from) if resume_from else None

  def initialize(self) -> None:
    """Initializes the algorithm, which in turn initializes the objective."""
    self.initial_params = self.algorithm.initialize(self.objective, self.seed)

  def run(self) -> torch.Tensor:
    """Runs the trainer and returns the final parameter vectors [population_size, param_size]."""
    start_time = time.time()
    self.logger.info(f'Started run for {self.epochs} epochs')
    self.objective.on_run_start(algorithm=self.algorithm)

    params = self.initial_params
    self.logger.info(f'Initialized {params.shape[0]} params with size {params.shape[1]}')

    # Test and save the initial params, unless resuming a run after a saved epoch.
    resumed = self._resume()
    if resumed is None:
      last_epoch = 0
      train_data = AlgorithmTrainData(params=params, best_param=params[0], stats={}, files={})
      test_data = self._test(0, start_time)
      self._save(0, train_data, test_data)
    else:
      last_epoch, elapsed, params = resumed
      start_time -= elapsed  # Elapsed-time columns continue from the resumed run's.

    for epoch in tqdm.trange(
      last_epoch + 1, self.epochs + 1, desc='epoch', disable=not self.show_progress
    ):
      try:
        train_data = self._train(epoch, start_time)
        test_data = self._test(epoch, start_time)
        self._save(epoch, train_data, test_data)
        params = train_data.params
      except BaseException:
        self.logger.exception(f'Epoch {epoch}: Error occurred')
        raise

    self.objective.on_run_end(algorithm=self.algorithm)
    run_time = time.time() - start_time
    self.logger.info(f'Completed run for {self.epochs} epochs in {run_time:.2f} seconds')

    return params

  def _resume(self) -> tuple[int, float, torch.Tensor] | None:
    """Restores the newest saved algorithm state (see `resume`) and the train/test rows up to it.

    Returns:
      The epoch of the restored state, the seconds elapsed up to it, and the parameter vectors saved
      with it, or None to start afresh.
    """
    if not self.resume and self.resume_from is None:
      return None
    source_dir = self.resume_from or self.logger.log_dir

    # Restore the newest readable algorithm state and its parameter vectors; a run killed while
    # saving can leave the newest files truncated.
    state_paths = sorted(
      (source_dir / 'algorithm').glob('*.pkl'),
      key=lambda path: int(path.stem),
      reverse=True,
    )
    for path in state_paths:
      try:
        state = utils.io.read_file(path)
        params = utils.io.read_file(source_dir / 'params' / path.name)
      except Exception as error:
        self.logger.info(f'Resume: skipping unreadable {path} ({error!r})')
        continue
      epoch = int(path.stem)
      self.algorithm.set_state(state)
      params = params.to(self.initial_params.device)
      break
    else:
      if self.resume == 'auto' and self.resume_from is None:
        return None
      raise FileNotFoundError(f'No readable algorithm state to resume from in {source_dir}')

    # Keep the train/test rows up to the restored epoch, as the following epochs are recomputed, and
    # read the seconds elapsed up to it. With `resume_from`, this run's log directory may not exist
    # yet.
    self.logger.log_dir.mkdir(parents=True, exist_ok=True)
    elapsed = 0.
    for phase in ('train', 'test'):
      source_path = source_dir / f'{phase}.csv'
      if not source_path.exists() or source_path.stat().st_size == 0:
        continue
      table = pd.read_csv(source_path, index_col=False)
      table = table[table[f'{phase}/epochs'] <= epoch]
      table.to_csv(self.logger.log_dir / f'{phase}.csv', index=False)
      self.logger.table(f'{phase}.csv').load()  # Later rows append to the kept rows.
      if len(table):
        elapsed = max(elapsed, float(table[f'{phase}/seconds'].iloc[-1]))

    self.logger.info(f'Resumed from {path}: continuing at epoch {epoch + 1}')
    return epoch, elapsed, params

  def _train(self, epoch: int, start_time: float) -> AlgorithmTrainData:
    """Runs a train step (every epoch)."""
    train_time = time.time()
    with torch.cuda.nvtx.range('objective_train_epoch_start'):
      self.objective.on_train_epoch_start(epoch=epoch, algorithm=self.algorithm)
    with torch.cuda.nvtx.range('algorithm_step'):
      data = self.algorithm.step(epoch, self.seed)
    with torch.cuda.nvtx.range('objective_train_epoch_end'):
      self.objective.on_train_epoch_end(epoch=epoch, algorithm=self.algorithm, train_data=data)

    # Save stats.
    curr_time = time.time()
    self.logger.table('train.csv').store({
      'train/epochs': epoch,
      'train/seconds': curr_time - start_time,
      'train/epoch_seconds': curr_time - train_time,
    }).store({
      f'train/{k}': v if isinstance(v, (int, float)) else v.numpy(force=True)
      for k, v in data.stats.items()
    }).save()
    self.logger.info(f'Epoch {epoch}: Completed train step')

    return data

  def _test(self, epoch: int, start_time: float) -> AlgorithmTestData | None:
    """Runs a test step (before first epoch, after every `test_freq` epochs, after last epoch)."""
    if (self.test_freq > 0 and epoch % self.test_freq == 0) or epoch == 0 or epoch == self.epochs:
      test_time = time.time()
      with torch.cuda.nvtx.range('objective_test_epoch_start'):
        self.objective.on_test_epoch_start(epoch=epoch, algorithm=self.algorithm)
      with torch.cuda.nvtx.range('algorithm_test'):
        data = self.algorithm.test(epoch, self.seed)
      with torch.cuda.nvtx.range('objective_test_epoch_end'):
        self.objective.on_test_epoch_end(epoch=epoch, algorithm=self.algorithm, test_data=data)

      # Save stats.
      curr_time = time.time()
      self.logger.table('test.csv').store({
        'test/epochs': epoch,
        'test/seconds': curr_time - start_time,
        'test/epoch_seconds': curr_time - test_time,
      }).store({
        f'test/{k}': v if isinstance(v, (int, float)) else v.numpy(force=True)
        for k, v in data.stats.items()
      }).save()
      self.logger.info(f'Epoch {epoch}: Completed test step')

      return data
    return None

  def _save(
    self,
    epoch: int,
    train_data: AlgorithmTrainData,
    test_data: AlgorithmTestData | None,
  ) -> None:
    """Saves parameters/artifacts (after every `save_freq` epochs, after last epoch)."""
    if (self.save_freq > 0 and epoch % self.save_freq == 0) or epoch == 0 or epoch == self.epochs:
      self.objective.on_save(
        epoch=epoch,
        algorithm=self.algorithm,
        train_data=train_data,
        test_data=test_data,
        log_dir=self.logger.log_dir,
      )
      params = train_data.params
      self.logger.file('params').save(f'{epoch}.pkl', params.cpu())
      self.logger.file('algorithm').save(f'{epoch}.pkl', self.algorithm.get_state())
      for k, v in train_data.files.items():
        self.logger.file('train', k).save(f'{epoch}.pkl', v)
      if test_data is not None:
        for k, v in test_data.files.items():
          self.logger.file('test', k).save(f'{epoch}.pkl', v)
      self.logger.info(f'Epoch {epoch}: Saved parameters and artifacts')
