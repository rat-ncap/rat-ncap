import dataclasses
import pathlib
import pickle
import pprint
import time
import typing as T

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import omegaconf as oc
import torch
import wadler_lindig as wl

import composer as cx
import composer.utils.gym as cxg
import composer.utils.rollout as cxr
import composer.utils.torch as cxt
import evolution2 as evolution
import tonic
import tonic.torch
import tonic.torch.agents as tta
import utils.logging as utlog

Agent = tta.ES

# ==================================================================================================
# Checkpoint

@dataclasses.dataclass
class MjxEnvObjectiveCheckpoint:
  """A saved model and what is needed to rebuild its environment and agent. The model holds the
  parameters that an epoch's test scored (after constraints): the same for every algorithm, e.g.
  the parameter point of a point-based ES or the best member of a population. `seed` seeds the
  rebuilt environment and agent; it is not the training seed."""
  epoch: int
  seed: int
  env_config: oc.DictConfig
  agent_config: oc.DictConfig
  subtask_space: cx.SubtaskSpace
  observation_space: cx.ObservationSpace
  action_space: cx.ActionSpace
  model_state_dict: dict[str, torch.Tensor]

  def initialize_env(self) -> tuple[cx.MjxEnv, cx.MjxEnvState]:
    """Returns the environment and its initial state of one world, seeded by `seed`."""
    env: cx.MjxEnv = hydra.utils.instantiate(self.env_config, _convert_='object')
    rng = jax.random.split(jax.random.key(self.seed), 1)
    state = env.initialize(rng)
    return env, state

  def initialize_agent(self) -> Agent:
    """Returns the agent, initialized for the spaces and loaded with the saved model."""
    agent: Agent = hydra.utils.instantiate(self.agent_config, _convert_='object')
    observation_space = cxg.to_gym_space(self.observation_space)
    action_space = cxg.to_gym_space(self.action_space)
    agent.initialize(observation_space, action_space, seed=self.seed)
    agent.model.load_state_dict(self.model_state_dict, strict=True)
    return agent


def save_checkpoint_spec(
  checkpoint_dir: pathlib.Path,
  *,
  env_config: oc.DictConfig,
  agent_config: oc.DictConfig,
  subtask_space: cx.SubtaskSpace,
  observation_space: cx.ObservationSpace,
  action_space: cx.ActionSpace,
) -> None:
  """Saves what a checkpoint directory needs besides the models to rebuild the environment and the
  agent: `env_config.yaml`, `agent_config.yaml` and the spaces of one world (`*_space.pkl`).

  Args:
    checkpoint_dir: Checkpoint directory, created if needed.
    env_config: Config of the environment.
    agent_config: Config of the agent.
    subtask_space: Subtask space of one world.
    observation_space: Observation space of one world.
    action_space: Action space of one world.
  """
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  oc.OmegaConf.save(env_config, checkpoint_dir / 'env_config.yaml')
  oc.OmegaConf.save(agent_config, checkpoint_dir / 'agent_config.yaml')
  with open(checkpoint_dir / 'subtask_space.pkl', 'wb') as f:
    pickle.dump(subtask_space, f)
  with open(checkpoint_dir / 'observation_space.pkl', 'wb') as f:
    pickle.dump(observation_space, f)
  with open(checkpoint_dir / 'action_space.pkl', 'wb') as f:
    pickle.dump(action_space, f)


def save_checkpoint(
  log_dir: pathlib.Path,
  *,
  epoch: int,
  env_config: oc.DictConfig,
  agent_config: oc.DictConfig,
  subtask_space: cx.SubtaskSpace,
  observation_space: cx.ObservationSpace,
  action_space: cx.ActionSpace,
  agent: Agent,
  epoch_name: str | None = None,
) -> None:
  """Saves a checkpoint to the log directory's `checkpoint` subdirectory: enough to rebuild both
  the environment and the agent.

  The configs and spaces (`save_checkpoint_spec`) are written only at epoch 0, which the trainer
  always tests. The model is written at every call, as `{epoch_name}.pt`, and its epoch as
  `last_epoch.pkl`.

  Args:
    log_dir: Log directory of the training run.
    epoch: Epoch of the model.
    env_config: Config of the environment.
    agent_config: Config of the agent.
    subtask_space: Subtask space of one world.
    observation_space: Observation space of one world.
    action_space: Action space of one world.
    agent: Agent whose model is saved.
    epoch_name: File name of the model (e.g. 'last'), or None for the epoch number.
  """
  epoch_name = epoch_name or str(epoch)
  checkpoint_dir = log_dir / 'checkpoint'
  checkpoint_dir.mkdir(exist_ok=True)
  if epoch == 0:
    save_checkpoint_spec(
      checkpoint_dir,
      env_config=env_config,
      agent_config=agent_config,
      subtask_space=subtask_space,
      observation_space=observation_space,
      action_space=action_space,
    )
  # (1) The `state_dict` saves all parameters and persistent buffers of the model. Unfortunately,
  # it does not efficiently handle tied parameters/buffers, so it saves duplicates of the same
  # tensor for each tied reference, making checkpoint sizes larger. However, `torch` does
  # not make it trivial to determine which buffers are persistent, and an approach that saves
  # `named_parameters` and `named_buffers` would be annoying to implement if we want the useful
  # `strict` error-checking that `load_state_dict` provides. Thus, we accept the tradeoff for now.
  # (2) Only the singleton model (loaded with the best parameters) is saved in the checkpoint.
  # (3) The model includes `actor`, `critic`, and all other submodules.
  torch.save(agent.model.state_dict(), checkpoint_dir / f'{epoch_name}.pt')
  with open(checkpoint_dir / 'last_epoch.pkl', 'wb') as f:
    pickle.dump(epoch, f)


def load_checkpoint(
  checkpoint_dir: pathlib.Path | str,
  *,
  epoch: T.Literal['first', 'last'] | int = 'last',
  seed: int = 0,
) -> MjxEnvObjectiveCheckpoint:
  """Loads a checkpoint from a checkpoint directory: a training run's `checkpoint` subdirectory, or
  a provided `checkpoints/<config>/seed<N>` directory, which have the same layout.

  Args:
    checkpoint_dir: Checkpoint directory.
    epoch: Epoch of the checkpoint: 'last' (the last tested epoch), 'first', or an epoch number
      saved by `checkpoint_freq`.
    seed: Seed used to initialize the environment and agent.

  Raises:
    FileNotFoundError: If the checkpoint files are missing.
    ValueError: If `epoch` is a string other than 'first' and 'last'.
  """
  checkpoint_dir = pathlib.Path(checkpoint_dir)

  # Resolve the checkpoint file of the epoch: `last.pt` holds the last tested epoch.
  if epoch == 'last' and (checkpoint_dir / 'last.pt').exists():
    epoch = T.cast(int, _load_pickle(checkpoint_dir / 'last_epoch.pkl'))
    epoch_name = 'last'
  elif isinstance(epoch, str):
    epochs = sorted(int(path.stem) for path in checkpoint_dir.glob('*.pt') if path.stem.isdigit())
    if not epochs:
      raise FileNotFoundError(f'No checkpoints found in: {checkpoint_dir}')
    if epoch not in ('first', 'last'):
      raise ValueError(f'Invalid checkpoint epoch: {epoch}')
    epoch = epochs[0] if epoch == 'first' else epochs[-1]
    epoch_name = str(epoch)
  else:
    epoch_name = str(epoch)

  return MjxEnvObjectiveCheckpoint(
    epoch=epoch,
    seed=seed,
    env_config=T.cast(oc.DictConfig, oc.OmegaConf.load(checkpoint_dir / 'env_config.yaml')),
    agent_config=T.cast(oc.DictConfig, oc.OmegaConf.load(checkpoint_dir / 'agent_config.yaml')),
    subtask_space=_load_pickle(checkpoint_dir / 'subtask_space.pkl'),
    observation_space=_load_pickle(checkpoint_dir / 'observation_space.pkl'),
    action_space=_load_pickle(checkpoint_dir / 'action_space.pkl'),
    model_state_dict=torch.load(
      checkpoint_dir / f'{epoch_name}.pt',
      map_location=torch.get_default_device(),
    ),
  )


def _load_pickle(path: pathlib.Path) -> T.Any:
  """Loads a pickled object."""
  with open(path, 'rb') as file:
    return pickle.load(file)


# ==================================================================================================
# Rollout

ParamsDict = dict[str, torch.Tensor]
BuffersDict = dict[str, torch.Tensor]


@dataclasses.dataclass
class MjxEnvObjectiveRollout:
  """A test rollout of the ensembled actor: its parameters, its buffers at each step, and the
  environment steps and actions, with a leading `[vectorize]` axis over the test environments."""
  params: ParamsDict  # w[0] = {name: param}
  buffers: list[BuffersDict]  # h[0:T] = [{name: buffer} for 0, ..., T]
  steps: list[cx.Step[cx.MjxEnvState]]  # s[0:T]
  actions: list[cx.Action]  # a[0:T-1]


def save_rollout(
  log_dir: pathlib.Path,
  *,
  epoch: int | T.Literal['last'],
  subtasks: torch.Tensor,
  params: ParamsDict,
  buffers: list[BuffersDict],
  steps: list[cx.Step[cx.MjxEnvState]],
  actions: list[cx.Action],
) -> None:
  """Saves a test rollout to the log directory's `rollout` subdirectory, one file per field.

  Args:
    log_dir: Log directory of the training run.
    epoch: Epoch of the rollout, or 'last'.
    subtasks: Subtask of each environment.
    params: Actor parameters w[0], [vectorize, ...] each.
    buffers: Actor buffers h[0:T] at each step 0, ..., T, [vectorize, ...] each.
    steps: Environment steps s[0:T].
    actions: Actions a[0:T-1].
  """
  rollout_dir = log_dir / 'rollout'
  rollout_dir.mkdir(exist_ok=True)
  (rollout_dir / 'subtasks').mkdir(exist_ok=True)
  (rollout_dir / 'params').mkdir(exist_ok=True)
  (rollout_dir / 'buffers').mkdir(exist_ok=True)
  (rollout_dir / 'steps').mkdir(exist_ok=True)
  (rollout_dir / 'actions').mkdir(exist_ok=True)
  # (1) Unlike in checkpoints, we save all `named_parameters` and `named_buffers` (persistent and
  # non-persistent) of the model since we want to capture the full agent state during the rollout.
  # These dicts do efficiently handle tied parameters/buffers. However, it means that we cannot use
  # `load_state_dict` with `strict` error-checking when re-loading a rollout into a model.
  # (2) These parameters/buffers are batched (e.g. `test_size` is batch dimension). Therefore, we
  # need to select a single index from the batch dimension when loading into a model for replay.
  # (3) These parameters include only `actor`.
  torch.save(
    subtasks,
    rollout_dir / 'subtasks' / f'{epoch}.pt',
  )
  torch.save(
    {k: v.cpu() for k, v in params.items()},
    rollout_dir / 'params' / f'{epoch}.pt',
  )
  torch.save(
    [{k: v.cpu() for k, v in buffer.items()} for buffer in buffers],
    rollout_dir / 'buffers' / f'{epoch}.pt',
  )
  cxr.save_steps(rollout_dir / 'steps' / f'{epoch}.pkl', steps)
  with open(rollout_dir / 'actions' / f'{epoch}.pkl', 'wb') as f:
    pickle.dump(actions, f)


def load_rollout(
  log_dir: pathlib.Path,
  *,
  epoch: int | str,
  env: cx.MjxEnv,
) -> MjxEnvObjectiveRollout:
  """Loads a rollout saved by `save_rollout`, without its subtasks. The steps' model is rebuilt from
  `env` (see `cxr.load_steps`)."""
  rollout_dir = log_dir / 'rollout'
  params: ParamsDict = torch.load(
    rollout_dir / 'params' / f'{epoch}.pt',
    map_location=torch.get_default_device(),
  )
  buffers: list[BuffersDict] = torch.load(
    rollout_dir / 'buffers' / f'{epoch}.pt',
    map_location=torch.get_default_device(),
  )
  steps = cxr.load_steps(rollout_dir / 'steps' / f'{epoch}.pkl', env)
  with open(rollout_dir / 'actions' / f'{epoch}.pkl', 'rb') as f:
    actions: list[cx.Action] = pickle.load(f)
  return MjxEnvObjectiveRollout(
    # subtasks=subtasks,
    params=params,
    buffers=buffers,
    steps=steps,
    actions=actions
  )


# ==================================================================================================
# Objective

class EnsembledAgent(tonic.torch.agent.Agent):
  """
  A utility class for ensembled evolution actor methods.
  EnsembledAgent.params and EnsembledAgent.buffers can be updated independently.
  The ensemble dimension of EnsembledAgent.params and EnsembledAgent.buffers must match,
  but can be changed between method calls.

  Usage:
  ```
  # Initialize ensembled evolution agent.
  agent_ensembled = EnsembledAgent(tonic.torch.agents.ES)
  agent_ensembled.initialize(observation_space, action_space, seed)

  # Initialize ensembled parameters and buffers.
  params, buffers = agent_ensembled.initial_state(vectorize)
  agent_ensembled.set_state(params, buffers)

  # Call ensembled agent methods.
  agent_ensembled.reset()
  action = agent_ensembled.step(observation)
  ```
  """
  def __init__(
    self,
    agent_builder: T.Callable[[], Agent],
    randomness: T.Literal['same', 'different'] = 'same',
  ):
    """
    Args:
      agent_builder: Builds an agent instance.
      randomness: Whether random ops (e.g. model noise) draw the same or different values across the
        ensemble dimension. See `torch.vmap`. We use `same` for evolutionary training algorithms
        so that parameters are evaluated under the same random conditions, and `different` for
        evaluation so that the ensemble is evaluated under different random conditions.
    """
    self.agent_builder = agent_builder
    self.randomness: T.Literal['same', 'different'] = randomness

    self.initialized = False

    self.agent = self.agent_builder()  # Singleton instance for params, buffer creation.
    self.params: dict[str, torch.Tensor]
    self.buffers: dict[str, torch.Tensor]

  def initial_state(
    self,
    vectorize: int,
    model_state_dict: dict[str, torch.Tensor] | None = None
  ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns `vectorize` stacked copies of the actor's parameters and buffers, [vectorize, ...]
    each, after loading `model_state_dict` (if given) into the singleton model."""
    assert self.initialized, 'EnsembledAgent must be initialized first.'
    if model_state_dict:
      self.agent.model.load_state_dict(model_state_dict)
    return torch.func.stack_module_state(
      [self.agent.model.actor for _ in range(vectorize)]  # type: ignore  # ActorLike not a Module
    )

  def set_state(
    self,
    params: dict[str, torch.Tensor] | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
  ):
    """Sets the given parameters and buffers of the ensemble, [ensemble, ...] each."""
    if params:
      self.params = params
    if buffers:
      self.buffers = buffers

  @T.override
  def initialize(
    self,
    observation_space: tonic.ObservationSpace,
    action_space: tonic.ActionSpace,
    seed: int,
  ) -> None:
    """Initializes the singleton model and builds the ensemble's vmapped actor functions."""
    super().initialize(observation_space, action_space, seed)
    self.agent.model.initialize(self.observation_space, self.action_space)

    # Make meta agent copy, removing concrete parameter/buffer values. This is useful for catching
    # errors arising from non-functional code patterns. The `functional_method` temporarily
    # replaces all parameters/buffers with the provided concrete values. Ops that reference tensors
    # other than these will error since they are on the 'meta' device. We must re-instantiate and
    # initialize the agent under the 'meta' device context, NOT invoke `to('meta')` on a deep copy
    # of the agent, since the latter approach breaks tied weights/references.
    with torch.device('meta'):
      self.agent_meta: Agent = self.agent_builder()
      self.agent_meta.model.initialize(observation_space, action_space)

    def model_reset(
      params: ParamsDict,
      buffers: BuffersDict,
      reset: torch.Tensor | None = None,
    ) -> BuffersDict:
      cxt.functional_method(
        self.agent_meta.model.actor,  # type: ignore (ActorLike is not a Module)
        (params, buffers),
        method=self.agent_meta.model.actor.reset,
        args=(reset,)
      )
      return buffers

    def model_constrain(
      params: ParamsDict,
      buffers: BuffersDict,
    ) -> ParamsDict:
      cxt.functional_method(
        self.agent_meta.model.actor,  # type: ignore (ActorLike is not a Module)
        (params, buffers),
        method=self.agent_meta.model.actor.constrainer.constrain,  # type: ignore
      )
      return params

    def agent_step(
      params: ParamsDict,
      buffers: BuffersDict,
      observations: tonic.Observation,
    ) -> tuple[tonic.Action, BuffersDict]:
      action = cxt.functional_method(
        self.agent_meta.model.actor,  # type: ignore (ActorLike is not a Module)
        (params, buffers),
        method=self.agent_meta.step,
        args=(observations,),
      )
      return action, buffers

    def agent_test_step(
      params: ParamsDict,
      buffers: BuffersDict,
      observations: tonic.Observation,
    ) -> tuple[tonic.Action, BuffersDict]:
      action = cxt.functional_method(
        self.agent_meta.model.actor,  # type: ignore (ActorLike is not a Module)
        (params, buffers),
        method=self.agent_meta.test_step,
        args=(observations,),
      )
      return action, buffers

    self.model_reset_batch: T.Callable[..., BuffersDict] = (
      torch.vmap(model_reset, randomness=self.randomness, out_dims=0)
    )
    self.model_constrain_batch: T.Callable[[ParamsDict, BuffersDict], ParamsDict] | None = (
      torch.vmap(model_constrain, randomness=self.randomness, out_dims=0)
      if hasattr(self.agent_meta.model.actor, 'constrainer') else None
    )
    self.agent_step_batch: T.Callable[
      [ParamsDict, BuffersDict, tonic.Observation],
      tuple[tonic.Action, BuffersDict],
    ] = (
      torch.vmap(agent_step, randomness=self.randomness, out_dims=(0, 0))
    )
    self.agent_test_step_batch: T.Callable[
      [ParamsDict, BuffersDict, tonic.Observation],
      tuple[tonic.Action, BuffersDict],
    ] = (
      torch.vmap(agent_test_step, randomness=self.randomness, out_dims=(0, 0))
    )
    self.initialized = True

  @T.override
  def step(self, observations: tonic.Observation) -> tonic.Action:
    """Returns each member's training action for its observations, advancing its buffers."""
    actions, self.buffers = self.agent_step_batch(self.params, self.buffers, observations)
    return actions

  @T.override
  def test_step(self, observations: tonic.Observation) -> tonic.Action:
    """Returns each member's test action for its observations, advancing its buffers."""
    actions, self.buffers = self.agent_test_step_batch(self.params, self.buffers, observations)
    return actions

  @T.override
  def record(
    self,
    observations: tonic.Observation,
    actions: tonic.Action,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    next_observations: tonic.Observation,
  ) -> None:
    raise NotImplementedError('EnsembledAgent is forward-only: record not supported.')

  @T.override
  def update(self) -> None:
    raise NotImplementedError('EnsembledAgent is forward-only: update not supported.')

  @T.override
  def save(self, path: str) -> None:
    raise NotImplementedError('EnsembledAgent is forward-only: save not supported.')

  @T.override
  def load(self, path: str) -> None:
    raise NotImplementedError('EnsembledAgent is forward-only: load not supported.')

  def reset(self, reset: torch.Tensor | None = None):
    """Resets the members' buffers (e.g. recurrent state): all, or those where `reset`,
    [ensemble], is true."""
    if reset is None:
      self.buffers = self.model_reset_batch(self.params, self.buffers)
    else:
      self.buffers = self.model_reset_batch(self.params, self.buffers, reset)

  def constrain(self):
    """Constrains the members' parameters with the actor's constrainer, if it has one."""
    if not self.model_constrain_batch:
      return
    self.params = self.model_constrain_batch(self.params, self.buffers)


class MjxEnvContext:
  """The environments and batched actor state of `MjxEnvObjective`'s train or test rollouts."""
  def __init__(
    self,
    env_config: oc.DictConfig,
    rollouts: int,
    timeouts: int,
    logger: utlog.Logger,
  ):
    """Initializes the context; `initialize` must be called before rolling out.

    Args:
      env_config: Config of the environment.
      rollouts: Rollouts per parameter vector, one per subtask.
      timeouts: Steps per rollout.
      logger: Logger.
    """
    self.env: cx.MjxEnv = hydra.utils.instantiate(env_config, _convert_='object')
    self.rollouts = rollouts
    self.timeouts = timeouts
    self.logger = logger
    self.params: ParamsDict
    self.buffers: BuffersDict

  def initialize(
    self,
    seed: int,
    batch_size: int,
    agent: EnsembledAgent,
  ):
    """Initializes `rollouts * batch_size` environments, their spaces and jitted reset and step,
    and the batched parameters and buffers of `agent` (initialized first if needed)."""
    # Initialize environment.
    self.logger.info('Initializing environment')
    self.batch_size = batch_size
    self.vectorize = self.rollouts * batch_size
    rng = jax.random.split(jax.random.key(seed), self.vectorize)
    state = self.env.initialize(rng)
    self.subtask_space = self.env.subtask_space(state).unvectorize()
    self.observation_space = self.env.observation_space(state).unvectorize()
    self.action_space = self.env.action_space(state).unvectorize()
    self.last_state: cx.MjxEnvState = state

    self.logger.info('Jitting environment reset and step')
    self.env_reset_batch = eqx.filter_jit(self.env.reset)
    self.env_step_batch = eqx.filter_jit(self.env.step)

    # Initialize model and batch its parameters/buffers. This main model aggregates normalizer
    # buffers across the batch.
    if not agent.initialized:
      self.logger.info('Initializing agent')
      observation_space = cxg.to_gym_space(self.observation_space)
      action_space = cxg.to_gym_space(self.action_space)
      agent.initialize(observation_space, action_space, seed)
    self.params, self.buffers = agent.initial_state(self.vectorize)


class MjxEnvObjective(evolution.Objective):
  """Objective that scores parameter vectors of a Tonic Torch agent's actor by rolling them out in
  vectorized MJX environments.

  Each parameter vector is rolled out once per subtask (`train_subtasks` or `test_subtasks`), for
  `train_timeouts` or `test_timeouts` steps. Its score is the mean reward per step, averaged over
  the subtasks. Training rollouts also update the agent (e.g. its observation normalizer); test
  rollouts are recorded for `on_save`, which checkpoints the first tested parameter vector.
  """
  def __init__(
    self,
    train_env_config: oc.DictConfig,
    test_env_config: oc.DictConfig,
    agent_config: oc.DictConfig,
    logger: utlog.Logger,
    *,
    subtasks: dict[str, T.Sequence[int | float]] | None = None,
    train_subtasks: list[int] | None = None,
    test_subtasks: list[int] | None = None,
    train_timeouts: int = 500,
    test_timeouts: int = 500,
    checkpoint_freq: int | None = None,
  ):
    """Initializes the objective; `initialize` must be called before scoring.

    Args:
      train_env_config: Config of the training environment.
      test_env_config: Config of the test environment.
      agent_config: Config of the agent (e.g. `tonic.torch.agents.ES`).
      logger: Logger.
      subtasks: Subtask values, {name: [value of each subtask]}, or None for one empty subtask.
      train_subtasks: Indices of the subtasks of the training rollouts, or None for all.
      test_subtasks: Indices of the subtasks of the test rollouts, or None for `train_subtasks`.
      train_timeouts: Steps per training rollout. Must equal the agent's `rollout_steps`.
      test_timeouts: Steps per test rollout.
      checkpoint_freq: Epochs between the checkpoints kept as `{epoch}.pt`, besides `last.pt`, or
        None to keep `last.pt` only.
    """
    self.train_env_config = train_env_config
    self.agent_config = agent_config
    self.logger = logger
    self.checkpoint_freq = checkpoint_freq

    self.subtasks = subtasks or {}
    self.subtask_count = len(next(iter(self.subtasks.values()))) if self.subtasks else 1
    train_subtasks = train_subtasks or list(range(self.subtask_count))
    test_subtasks = test_subtasks or train_subtasks
    self.train_subtasks = train_subtasks
    self.test_subtasks = test_subtasks

    # `train_rollouts` is set by the number of subtasks, which must be consistent between
    # `train_subtasks` and all entries of `curriculum`.

    # Instantiate from configs since `env`/`model` objects are not serializable (i.e. cannot be
    # pickled and sent to workers). Previously, pickling the builder functions pulled in imports
    # which breaks when code structure changes.
    self.train_context = MjxEnvContext(
      train_env_config,
      len(train_subtasks),
      train_timeouts,
      logger,
    )
    self.test_context = MjxEnvContext(
      test_env_config,
      len(test_subtasks),
      test_timeouts,
      logger,
    )
    self.agent_singleton: Agent = hydra.utils.instantiate(agent_config, _convert_='object')
    self.agent_ensembled = EnsembledAgent(
      lambda: hydra.utils.instantiate(agent_config, _convert_='object')
    )

    assert train_timeouts == self.agent_singleton.rollout_steps, (
      'Train timeouts must match agent rollout steps.'
    )

  def __pdoc__(self, **kwargs):
    """Returns the `wadler_lindig` document of the objective: its environment, agent, rollouts per
    parameter vector and steps per rollout."""
    return wl.ConcatDoc(
      wl.TextDoc(f'{self.__class__.__name__}'),
      wl.TextDoc('('),
      wl.NestDoc(
        wl.ConcatDoc(
          wl.BreakDoc(''),
          wl.GroupDoc(
            wl.join(
              wl.comma,
              [
                wl.ConcatDoc(wl.TextDoc('env='), wl.pdoc(self.train_context.env)),
                wl.ConcatDoc(
                  wl.TextDoc('agent='),
                  wl.NestDoc(
                    wl.ConcatDoc(wl.BreakDoc(''), wl.pdoc(self.agent_singleton)),
                    indent=kwargs['indent'],
                  )
                ),
                wl.ConcatDoc(wl.TextDoc('rollouts='), wl.pdoc(self.train_context.rollouts)),
                wl.ConcatDoc(wl.TextDoc('timeouts='), wl.pdoc(self.train_context.timeouts)),
              ],
            )
          )
        ),
        indent=kwargs['indent'],
      ),
      wl.BreakDoc(''),
      wl.TextDoc(')'),
    )

  def __repr__(self) -> str:
    return wl.pformat(self, width=100)

  @T.override
  def initialize(self, seed: int, train_size: int, test_size: int = 1):
    """Initializes the train and test environments for `train_size` and `test_size` parameter
    vectors, and the agents."""
    self.train_context.initialize(seed, train_size, self.agent_ensembled)
    self.test_context.initialize(seed, test_size, self.agent_ensembled)
    self.observation_space = cxg.to_gym_space(self.train_context.observation_space)
    self.action_space = cxg.to_gym_space(self.train_context.action_space)
    self.agent_singleton.initialize(self.observation_space, self.action_space, seed)

  @T.override
  def on_run_start(self, *, algorithm: evolution.Algorithm):
    """Logs the environment, the spaces, the model, and the actor's parameter and buffer shapes."""
    self.logger.info(f'env:\n{self.train_context.env}')
    self.logger.info(f'env.observation_space:\n{self.observation_space}')
    self.logger.info(f'env.action_space:\n{self.action_space}')
    self.logger.info(f'model:\n{self.agent_singleton.model}')
    self.logger.info(
      f'model.actor.params:\n{pprint.pformat(
        {k: tuple(v.shape) for k, v in self.train_context.params.items()},
        indent=2, width=100, sort_dicts=False)}'
    )
    self.logger.info(
      f'model.actor.buffers:\n{pprint.pformat(
        {k: tuple(v.shape) for k, v in self.train_context.buffers.items()},
        indent=2, width=100, sort_dicts=False)}'
    )

  @T.override
  def on_train_epoch_end(
    self,
    *,
    epoch: int,
    algorithm: evolution.Algorithm,
    train_data: evolution.AlgorithmTrainData,
  ):
    """Copies the singleton actor's buffers (e.g. the normalizer statistics that training updates)
    into the batched buffers of the train and test actors. This also copies its state buffers
    (e.g. recurrent state), which `get_score` resets before each rollout."""
    for name, buffer in self.agent_singleton.model.actor.named_buffers():  # type: ignore
      self.train_context.buffers[name][:] = buffer
      self.test_context.buffers[name][:] = buffer

  @T.override
  def on_save(
    self,
    *,
    epoch: int,
    algorithm: evolution.Algorithm,
    train_data: evolution.AlgorithmTrainData,
    test_data: evolution.AlgorithmTestData | None,
    log_dir: pathlib.Path,
  ):
    """Saves a tested epoch's checkpoint, of the first tested parameter vector, and test rollout:
    as 'last', and every `checkpoint_freq` epochs also as the epoch number."""
    if not test_data:
      return

    subtask_space = test_data.files.pop('subtask_space')
    observation_space = test_data.files.pop('observation_space')
    action_space = test_data.files.pop('action_space')

    subtasks = test_data.files.pop('subtasks')
    params = test_data.files.pop('params')

    # Load the main model with the parameters that the test scored (constrained) to checkpoint it:
    # those of the first test environment, i.e. the algorithm's first test parameter vector.
    with torch.no_grad():
      for name, param in self.agent_singleton.model.actor.named_parameters():  # type: ignore
        param.copy_(params[name][0])
    buffers = test_data.files.pop('buffers')
    steps = test_data.files.pop('steps')
    actions = test_data.files.pop('actions')

    epochs: list[int | T.Literal['last']] = ['last']
    if self.checkpoint_freq and epoch % self.checkpoint_freq == 0:
      epochs.append(epoch)
    for e in epochs:
      save_checkpoint(
        log_dir=log_dir,
        epoch=epoch,
        env_config=self.train_env_config,
        agent_config=self.agent_config,
        subtask_space=subtask_space,
        observation_space=observation_space,
        action_space=action_space,
        agent=self.agent_singleton,
        epoch_name=str(e),
      )
      save_rollout(
        log_dir=log_dir,
        epoch=e,
        subtasks=subtasks,
        params=params,
        buffers=buffers,
        steps=steps,
        actions=actions,
      )

  @T.override
  def get_initial_params(self, seeds: torch.Tensor) -> torch.Tensor:
    """Returns the actor's trainable parameters as initialized with each seed, [seeds, size]."""
    initial_params: list[torch.Tensor] = []
    with torch.no_grad():
      for seed in seeds:  # [seed_size]
        # Initialize dummy models using different seeds to get trainable parameters.
        torch.manual_seed(seed.item())
        agent: Agent = hydra.utils.instantiate(self.agent_config, _convert_='object')
        agent.initialize(self.observation_space, self.action_space, int(seed.item()))
        param = cxt.pack_tensors(
          dict(agent.model.actor.named_parameters()),  # type: ignore
          grad_only=True,
        )
        initial_params.append(param)
    assert all(p.shape == initial_params[0].shape for p in initial_params)
    return torch.stack(initial_params, dim=0)  # [batch_size, param_size]

  @T.override
  def get_updated_params(self, params: torch.Tensor) -> torch.Tensor:
    """Returns the parameter vectors constrained by the actor's constrainer, if it has one. The
    first vectors of the train context's batched parameters are overwritten."""
    if self.agent_ensembled.model_constrain_batch is None: return params
    with torch.no_grad():
      # Number of params may be smaller than `self.vectorize`, so overwrite the first `size` idxs of
      # the batched params, and ignore the rest.
      size = params.shape[0]
      params_dict = cxt.unpack_tensors(
        params,
        self.train_context.params,
        grad_only=True,
        batch_idxs=[slice(0, size)],
      )
      buffers_dict = self.train_context.buffers
      params_dict = self.agent_ensembled.model_constrain_batch(params_dict, buffers_dict)
      updated_params = cxt.pack_tensors(params_dict, grad_only=True, batch_dims=1)[:size]
    return updated_params

  def reset_agent(self, context: MjxEnvContext, params: ParamsDict) -> None:
    """Loads the params of a rollout into the ensembled agent, constrained (evolution children have
    unconstrained mutations), with reset buffers since the params changed."""
    self.agent_ensembled.set_state(params=params, buffers=context.buffers)
    self.agent_ensembled.constrain()
    self.agent_ensembled.reset()

  @T.override
  def get_score(
    self,
    params: torch.Tensor,  # [batch_size, param_size]
    seeds: torch.Tensor,  # [batch_size]
    *,
    mode: T.Literal['train', 'test'],
    epoch: int = 0,
    stats: dict[str, float | torch.Tensor],
    files: dict[str, T.Any],
  ) -> torch.Tensor:  # [batch_size]
    """Returns each parameter vector's score (see the class), [batch_size]."""
    match mode:
      # --------------------------------------------------------------------------------------------
      case 'train':
        # In a train step, the full population of parameters is rolled out in the train environment.
        # The agent is run in train mode and is updated at the end of the step, buffers are updated.
        context = self.train_context
        agent_step = self.agent_ensembled.step
        subtask_indices = self.train_subtasks

        def record_first(
          params: ParamsDict,
          buffers: BuffersDict,
          step: cx.Step[cx.MjxEnvState],
        ):
          pass  # No-op in train mode.

        def record(
          observations: cxt.FlatTensorDict,
          actions: cxt.FlatTensorDict,
          step: cx.Step,
          buffers: BuffersDict,
        ):
          self.agent_singleton.record(
            observations=observations,  # type: ignore
            actions=actions,  # type: ignore
            rewards=cxt.jax_to_torch(step.reward),
            terminations=cxt.jax_to_torch(step.terminated),
            truncations=cxt.jax_to_torch(step.truncated),
            next_observations=cxt.jax_to_torch(step.observation),  # type: ignore
          )

        def update(average_return: torch.Tensor) -> tuple[torch.Tensor, dict[str, T.Any]]:
          metrics = self.agent_singleton.update()
          # [vectorize] -> [batch_size, rollouts] -> [batch_size]
          rollout_scores = metrics['scores'].reshape((context.batch_size, context.rollouts))
          return rollout_scores.mean(dim=1), dict(metrics)

      # --------------------------------------------------------------------------------------------
      case 'test':
        # In a test step, a subset of parameters selected by the algorithm are rolled out in the
        # test environment. The step is a no-op to any state related to training.
        context = self.test_context
        agent_step = self.agent_ensembled.test_step
        subtask_indices = self.test_subtasks

        def record_first(
          params: ParamsDict,
          buffers: BuffersDict,
          step: cx.Step[cx.MjxEnvState],
        ):
          files['subtask_space'] = context.subtask_space
          files['observation_space'] = context.observation_space
          files['action_space'] = context.action_space

          files['subtasks'] = cxt.jax_to_torch(subtasks)  # constant, [vectorize]
          files['params'] = params  # constant, w[0:T]
          files['buffers'] = [{k: v.clone() for k, v in buffers.items()}]  # h[0]
          files['steps'] = [step]  # s[0]
          files['actions'] = []  # empty

        def record(
          observations: cxt.FlatTensorDict,
          actions: cxt.FlatTensorDict,
          step: cx.Step,
          buffers: BuffersDict,
        ):
          files['actions'].append(cxt.torch_to_jax(actions))  # a[t]
          files['steps'].append(step)  # s[t+1]
          files['buffers'].append({k: v.clone() for k, v in buffers.items()})  # h[t+1]

        def update(average_return: torch.Tensor) -> tuple[torch.Tensor, dict[str, T.Any]]:
          return average_return, {}

    # ----------------------------------------------------------------------------------------------

    with torch.no_grad():
      # Prepare context-dependent rollout settings.
      vectorize = context.vectorize
      timeouts = context.timeouts
      rollouts = context.rollouts
      subtasks = {k: jnp.array(v)[jnp.array(subtask_indices)] for k, v in self.subtasks.items()}
      subtasks = cx.tile_value(
        subtasks,
        context.batch_size,
      )

      # Unpack step functions.
      agent_step = T.cast(T.Callable[[cxt.FlatTensorDict], cxt.FlatTensorDict], agent_step)
      env_reset = context.env_reset_batch
      env_step = context.env_step_batch

      # Load the evolution children params (grad=True), merged with the model's other parameters,
      # into the model, constrained and with reset buffers. Use the global seed due to torch.
      params_pack = params.repeat_interleave(rollouts, dim=0)  # [batch_size * rollouts, param_size]
      torch.manual_seed(seeds[0].item())
      self.reset_agent(context, cxt.unpack_tensors(params_pack, context.params, grad_only=True))

      # Reset environment, with a key per environment.
      rng = jax.vmap(
        lambda seed: jax.random.split(jax.random.key(seed), rollouts)  # [rollouts]
      )(cxt.torch_to_jax(seeds))  # [batch_size, rollouts]
      rng = rng.flatten()  # [vectorize] = [batch_size * rollouts]
      with torch.cuda.nvtx.range('rollout_reset_0'):
        step: cx.Step[cx.MjxEnvState] = env_reset(context.last_state, rng, subtask=subtasks)

      # Step agent and environment for `T = timeouts` steps, recording data for updates and
      # monitoring.
      rollout_time = time.perf_counter()
      returns = torch.zeros(vectorize)  # [vectorize]
      rollout_steps = torch.zeros(vectorize)  # [vectorize]

      # w[0], h[0], s[0]
      record_first(self.agent_ensembled.params, self.agent_ensembled.buffers, step)

      for t in range(timeouts):
        with torch.cuda.nvtx.range(f'rollout_step_{t}'):
          observations = cxt.jax_to_torch(step.observation)
          actions = agent_step(observations)
          action = cxt.torch_to_jax(actions)
          step = env_step(step.state, action)

          # o[t], a[t], s[t+1], h[t+1]
          record(observations, actions, step, self.agent_ensembled.buffers)

          returns += cxt.jax_to_torch(step.reward) / timeouts  # r[t] = reward(s[t+1])
          rollout_steps += cxt.jax_to_torch(~step.terminated)  # e[t] = terminated(s[t+1])

      context.last_state = step.state

      rollout_steps = rollout_steps.reshape((-1, rollouts))  # [batch_size, rollouts]
      returns = returns.reshape((-1, rollouts))  # [batch_size, rollouts]
      total_return = returns.mean(dim=1)  # [batch_size]
      subtask_returns = {subtask: [] for subtask in range(self.subtask_count)}
      for subtask, subtask_return in zip(subtask_indices, returns.mean(dim=0)):  # [rollouts]
        subtask_returns[subtask].append(subtask_return.item())
      rollout_time = time.perf_counter() - rollout_time

      # Update agent: (1) Perform updates on shared model params/buffers (e.g. critic params,
      # normalizer buffers). (2) Return test/train-mode-dependend scores and metrics.
      rollout_scores, metrics = update(total_return)

      # Update `stats` table with data for monitoring.
      stats['total_return'] = total_return
      for subtask, subtask_return in subtask_returns.items():
        stats[f'subtask_return/{subtask}'] = torch.tensor(subtask_return).mean().item()
      stats['rollout_steps'] = rollout_steps
      stats['rollout_time'] = rollout_time
      stats.update(metrics)

    return rollout_scores
