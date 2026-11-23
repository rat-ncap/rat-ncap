"""
Policies and record functions for testing trained checkpoints with `test.py`.
"""
import fnmatch
import os
import pathlib
import typing as T

import hydra
import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
import torch

import circuits as cc
import composer as cx
import composer.utils.gym as cxg
import composer.utils.rollout as cxr
import composer.utils.torch as cxt
import mjx_objective as mjxo
import models
import utils.string as uts

# The repository root, against which relative checkpoint paths are resolved.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class PolicyState(T.TypedDict):
  """State of an `EnsembledPolicy`: the batched parameters and buffers of its specimens."""
  params: dict[str, torch.Tensor]
  buffers: dict[str, torch.Tensor]


# ==================================================================================================
# Policies

class EnsembledPolicy(cxr.Policy[PolicyState]):
  """Runs a single checkpoint as a vectorized ensemble of `population` specimens, where each
  specimen has its own copy of the parameters and buffers and draws independent model noise.
  Environments are laid out population-major (see `composer.tile_value`), so specimen `p` occupies
  environments `[p * subtasks, (p + 1) * subtasks)`."""
  def __init__(
    self,
    checkpoint: os.PathLike | str | mjxo.MjxEnvObjectiveCheckpoint,
    population: int = 1,
    epoch: T.Literal['first', 'last'] | int = 'last',
    seed: int = 0,
  ):
    """
    Args:
      checkpoint: A checkpoint directory (see `mjx_objective.load_checkpoint`; a relative path is
        relative to the repository root, e.g. 'checkpoints/1_4_flxext/seed1'), or a loaded
        checkpoint.
      population: Number of specimens.
      epoch: Epoch of the checkpoint to load from a directory.
      seed: Seed of the agent rebuilt from a directory.
    """
    if not isinstance(checkpoint, mjxo.MjxEnvObjectiveCheckpoint):
      checkpoint = mjxo.load_checkpoint(REPO_ROOT / checkpoint, epoch=epoch, seed=seed)
    self.checkpoint = checkpoint
    self.population = population

    agent_config = self.checkpoint.agent_config
    # Draw independent model noise (e.g. RG oscillator noise) for each specimen.
    self.agent = mjxo.EnsembledAgent(
      lambda: hydra.utils.instantiate(agent_config, _convert_='object'),
      randomness='different',
    )
    self.agent.initialize(
      cxg.to_gym_space(self.checkpoint.observation_space),
      cxg.to_gym_space(self.checkpoint.action_space),
      self.checkpoint.seed,
    )

  @T.override
  def initialize(
    self,
    observation_space: cx.ObservationSpace,
    action_space: cx.ActionSpace,
    rng: jax.Array,
  ) -> None:
    """Checks that the environment's spaces are the checkpoint's.

    Raises:
      ValueError: If the observation or action space differs from the checkpoint's (e.g. for a
        different body config).
    """
    checkpoint = self.checkpoint
    observation_space_one = cxg.to_gym_space(observation_space.unvectorize())
    if observation_space_one != cxg.to_gym_space(checkpoint.observation_space):
      raise ValueError(
        f'Environment observation space does not match checkpoint:\n'
        f'{observation_space.unvectorize()}\n{checkpoint.observation_space}'
      )
    action_space_one = cxg.to_gym_space(action_space.unvectorize())
    if action_space_one != cxg.to_gym_space(checkpoint.action_space):
      raise ValueError(
        f'Environment action space does not match checkpoint:\n'
        f'{action_space.unvectorize()}\n{checkpoint.action_space}'
      )

  @T.override
  def reset(self, rng: jax.Array):
    """Rebuilds the ensemble from the checkpoint for `rng.shape[0]` environments, modifies it
    (`modify_state`) and resets it. Seeds torch's global random state, which draws the model noise,
    from `rng[0]`.

    Raises:
      ValueError: If the number of environments is not a multiple of `population`.
    """
    vectorize = rng.shape[0]
    if vectorize % self.population != 0:
      raise ValueError(
        f'Vectorize ({vectorize}) must be a multiple of population ({self.population}).'
      )
    # Seed model noise from the rollout seed for reproducibility.
    torch.manual_seed(int(jax.random.randint(rng[0], (), 0, jnp.iinfo(jnp.int32).max)))
    with torch.no_grad():
      params, buffers = self.agent.initial_state(
        vectorize,
        model_state_dict=self.checkpoint.model_state_dict,
      )
      self.agent.set_state(params=params, buffers=buffers)
      member = torch.arange(vectorize) // (vectorize // self.population)  # Specimen of each env.
      self.modify_state(member, rng)
      self.agent.reset()

  def modify_state(self, member: torch.Tensor, rng: jax.Array) -> None:
    """Hook to modify the ensemble state after it is rebuilt from the checkpoint on every reset.

    Args:
      member: Specimen index of each environment, [vectorize].
      rng: Rollout random keys, [vectorize].
    """

  @T.override
  def step(self, observation: cx.Observation, rng: jax.Array) -> cx.Action:
    """Returns each environment's action from its specimen's test step. `rng` is unused: the
    model noise is drawn from torch's global random state."""
    with torch.no_grad():
      observations = T.cast(
        T.OrderedDict[str, torch.Tensor],
        cxt.flatten_tensors(cxt.jax_to_torch(observation)),
      )
      actions = T.cast(dict[str, torch.Tensor], self.agent.test_step(observations))
      return cxt.torch_to_jax(actions)

  @T.override
  def get_state(self) -> PolicyState:
    """Returns the ensemble's parameters and buffers: the tensors themselves, not copies."""
    return PolicyState(params=self.agent.params, buffers=self.agent.buffers)

  @T.override
  def set_state(self, state: PolicyState) -> None:
    """Sets the ensemble's parameters and buffers."""
    self.agent.set_state(params=state['params'], buffers=state['buffers'])


class AblatedPolicy(EnsembledPolicy):
  """Scales the activation of selected rhythm generation neurons by `weight` (0 = full ablation).
  The neurons are patched once, at construction, so every rollout of the policy is ablated."""
  def __init__(
    self,
    *,
    neurons: T.Sequence[str],
    weight: float,
    **kwargs,
  ):
    """
    Args:
      neurons: Names of rhythm generation neurons to ablate (e.g. 'diagd_v0v_FL').
      weight: Activation scale applied to all `neurons`.
      **kwargs: Arguments passed to `EnsembledPolicy`.
    """
    super().__init__(**kwargs)
    self.neurons = neurons
    self.weight = weight

    # Patch the meta agent, since its modules execute the forward pass in `EnsembledAgent`.
    actor = T.cast(models.RatNcapActor, self.agent.agent_meta.model.actor)
    rg = T.cast(models.RhythmGenerationCircuit, T.cast(cc.Simulator, actor.sim).rhythm_generation)
    for name in neurons:
      neuron: cc.Basic = getattr(rg, name)
      neuron.activation = lambda v, activation=neuron.activation: activation(v) * weight


class PerturbedPolicy(EnsembledPolicy):
  """Perturbs the parameters of each specimen with independent Gaussian noise. Noise is
  resampled every rollout, and the actor's constrainer (if any) is applied after perturbation."""
  def __init__(
    self,
    *,
    mode: T.Literal['additive', 'multiplicative'],
    sigma: float,
    include: T.Sequence[str] = ('sim.*',),
    exclude: T.Sequence[str] = ('*.bias_*',),
    constrain_sign: bool = True,
    **kwargs,
  ):
    """
    Args:
      mode: Whether noise is added (`p + sigma * n`) or multiplied (`p * (1 + sigma * n)`).
      sigma: Noise standard deviation.
      include: Glob patterns of actor parameter names to perturb.
      exclude: Glob patterns of actor parameter names to not perturb, taking precedence.
      constrain_sign: Whether to prevent parameters from changing sign.
      **kwargs: Arguments passed to `EnsembledPolicy`.
    """
    super().__init__(**kwargs)
    if mode not in ('additive', 'multiplicative'):
      raise ValueError(f'Unsupported mode: {mode}')
    self.mode = mode
    self.sigma = sigma
    self.include = include
    self.exclude = exclude
    self.constrain_sign = constrain_sign

  @T.override
  def modify_state(self, member: torch.Tensor, rng: jax.Array) -> None:
    """Perturbs the included parameters with noise drawn per specimen from `rng[0]`, and constrains
    them."""
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(int(jax.random.randint(rng[0], (), 0, jnp.iinfo(jnp.int32).max)))

    params = dict(self.agent.params)
    for name, param in params.items():
      if not uts.in_glob_range(self.include, name) or uts.in_glob_range(self.exclude, name):
        continue
      # Sample noise per specimen, shared across that specimen's subtasks.
      noise = torch.randn(
        (self.population, *param.shape[1:]),
        generator=generator,
        device=param.device,
        dtype=param.dtype,
      )[member]
      match self.mode:
        case 'additive':
          perturbed = param + self.sigma * noise
        case 'multiplicative':
          perturbed = param * (1 + self.sigma * noise)
      if self.constrain_sign:
        perturbed = torch.where(param > 0, perturbed.clamp(min=0), perturbed)
        perturbed = torch.where(param < 0, perturbed.clamp(max=0), perturbed)
      params[name] = perturbed
    self.agent.set_state(params=params)

    if getattr(self.agent.agent.model.actor, 'constrainer', None) is not None:
      self.agent.constrain()


# ==================================================================================================
# Step hooks

class ScheduledObservation:
  """Rollout `on_reset`/`on_step` hook that overrides an observation (e.g. `target_vx`) with a
  piecewise-constant schedule over elapsed time, so the policy sees a time-varying command while the
  task's subtask (and therefore its rewards) stays fixed."""
  def __init__(self, *, key: str, schedule: T.Sequence[tuple[float, float]]):
    """
    Args:
      key: Observation key to override.
      schedule: `(start_time, value)` pairs in seconds; each value holds until the next start time.
        Times before the first start time use the first value.
    """
    schedule = sorted(schedule)
    self.key = key
    self.times = jnp.array([t for t, _ in schedule])
    self.values = jnp.array([v for _, v in schedule])

  def __call__(self, step: cx.Step[cx.MjxEnvState]) -> cx.Step[cx.MjxEnvState]:
    """Returns the step with the observation of `key` set to the schedule's value at the step's
    time. The step's observation is modified in place."""
    time = step.state.steps * step.state.env_timestep  # [vectorize]
    index = jnp.clip(jnp.searchsorted(self.times, time, side='right') - 1, 0)
    observation = step.observation[self.key]
    value = self.values[index].reshape(observation.shape).astype(observation.dtype)
    step.observation[self.key] = value
    return step


# ==================================================================================================
# Record functions

def record_terminated(ctx: cxr.RecordContext) -> np.ndarray:
  """Records whether each environment's episode has ended, [vectorize]."""
  return np.array(ctx.step.terminated)


def record_observation(ctx: cxr.RecordContext, *, key: str) -> np.ndarray:
  """Records the observation of `key`, [vectorize, ...]."""
  return np.array(ctx.step.observation[key])


def record_speed(ctx: cxr.RecordContext) -> np.ndarray:
  """Records the root forward velocity (m/s) in the global frame, [vectorize]."""
  if 'get_root_global_linvel' not in ctx.extras:
    body = ctx.env.task.children['body']
    ctx.extras['get_root_global_linvel'] = cx.vmap_state(
      T.cast(T.Callable[[cx.MjxEnvState], jax.Array], body.get_root_global_linvel),
      jit=True,
    )
  get_root_global_linvel = ctx.extras['get_root_global_linvel']
  return np.array(get_root_global_linvel(ctx.step.state)[:, 0])


def record_body(ctx: cxr.RecordContext, *, getter: str) -> np.ndarray:
  """Records the value of a body getter (e.g. `get_foot_global_pos`), [vectorize, ...]."""
  key = f'record_body/{getter}'
  if key not in ctx.extras:
    body = ctx.env.task.children['body']
    ctx.extras[key] = cx.vmap_state(
      T.cast(T.Callable[[cx.MjxEnvState], jax.Array], getattr(body, getter)),
      jit=True,
    )
  return np.array(ctx.extras[key](ctx.step.state))


def record_limb_tactile(ctx: cxr.RecordContext) -> np.ndarray:
  """Records each limb's summed tactile observation, [vectorize, limbs (FL, FR, HL, HR)]."""
  limbs = ('FL', 'FR', 'HL', 'HR')
  return np.stack([
    np.asarray(ctx.step.observation[f'body/tactile_limb{limb}']).sum(-1) for limb in limbs
  ], axis=-1)


def record_rhythm_generation_units(ctx: cxr.RecordContext[PolicyState]) -> dict[str, np.ndarray]:
  """Records the rhythm generation time (s), [vectorize], and the value of every unit (interneurons
  included), `{unit: [vectorize]}`, from the state of an `EnsembledPolicy`."""
  buffers = ctx.policy_state['buffers']
  prefix, suffix = 'sim.rhythm_generation.', '.value'
  return {
    'time': buffers['sim.time'][:, 0].cpu().numpy() / 1000,  # Convert ms to s.
    **{
      name[len(prefix):-len(suffix)]: buffers[name][:, 0].cpu().numpy()
      # The rhythm generator's own units ('sim.rhythm_generation.<unit>.value'), not nested ones.
      for name in fnmatch.filter(buffers, f'{prefix}*{suffix}') if name.count('.') == 3
    },
  }


def record_observations(ctx: cxr.RecordContext) -> dict[str, np.ndarray]:
  """Records every policy observation as the policy sees it (after the body's rescaling),
  `{key: [vectorize, ...]}`."""
  return {k: np.array(v) for k, v in ctx.step.observation.items()}


def record_actions(ctx: cxr.RecordContext) -> dict[str, np.ndarray]:
  """Records every action of the step (e.g. the flexor-extensor interface's `body/muscle_flx_*`
  and `body/muscle_ext_*`, before the controller's clipping and filtering),
  `{key: [vectorize, ...]}`."""
  return {k: np.array(v) for k, v in ctx.action.items()}


def record_reward_terms(ctx: cxr.RecordContext) -> dict[str, np.ndarray]:
  """Records the task's reward terms from the step's metrics (`reward/weight/<term>` and
  `reward/value/<term>`; the reward is the sum of weight x value), `{'weight/<term>' or
  'value/<term>': [vectorize]}`."""
  return {
    k.removeprefix('reward/'): np.array(v)
    for k, v in ctx.step.metrics.items() if k.startswith('reward/')
  }


def record_rhythm_generation(ctx: cxr.RecordContext[PolicyState]) -> dict[str, np.ndarray]:
  """Records the rhythm generation time (s), [vectorize], and output unit values,
  [vectorize, units], from the state of an `EnsembledPolicy`."""
  policy = T.cast(EnsembledPolicy, ctx.policy)
  actor = T.cast(models.RatNcapActor, policy.agent.agent.model.actor)
  rg = T.cast(models.RhythmGenerationCircuit, T.cast(cc.Simulator, actor.sim).rhythm_generation)
  buffers = ctx.policy_state['buffers']
  return {
    'time': buffers['sim.time'][:, 0].cpu().numpy() / 1000,  # Convert ms to s.
    'outputs': torch.cat([
      buffers[f'sim.rhythm_generation.{unit}.value'] for unit in rg.get_output_units()
    ], dim=-1).cpu().numpy(),
  }


def record_joint_side_profiles(
  joint_groups: T.Mapping[str, T.Sequence[str]],
  root_body: str = 'torso',
) -> cxr.RecordFn:
  """Returns a record function of joint side profiles: the coordinates of the root body and the
  anchors of each group's joints in the side-profile plane, which is spanned by the root body's
  horizontal heading and the vertical axis.

  Args:
    joint_groups: `{group_name: [joint_name]}` joints whose anchors to record.
    root_body: Name of the MuJoCo body whose x-axis gives the heading.

  Returns:
    Record function producing `{group_name: [vectorize, 1 + n, 2]}` (forward, vertical)
    coordinates of the root body followed by the group's `n` joint anchors.
  """
  def record(ctx: cxr.RecordContext) -> dict[str, np.ndarray]:
    state = ctx.step.state
    key = ('joint_side_profiles', id(record))
    if key not in ctx.extras:
      mj_model = state.mj_model
      root_body_id = mj.mj_name2id(mj_model, int(mj.mjtObj.mjOBJ_BODY), root_body)
      joint_ids = {
        group: np.array([mj.mj_name2id(mj_model, int(mj.mjtObj.mjOBJ_JOINT), j) for j in joints])
        for group, joints in joint_groups.items()
      }
      ctx.extras[key] = (root_body_id, joint_ids)
    root_body_id, joint_ids = ctx.extras[key]

    mjx_data = state.mjx_data
    root_position = np.asarray(mjx_data.xpos[:, root_body_id])  # [b, 3]
    root_x_axis = np.asarray(mjx_data.xmat[:, root_body_id, :, 0])  # [b, 3]
    forward_axis = root_x_axis * np.array([1, 1, 0])  # [b, 3]
    forward_axis_norm_sq = np.maximum((forward_axis**2).sum(-1), 1e-8)  # [b]
    joint_positions = np.asarray(mjx_data.xanchor)  # [b, joints, 3]
    side_profiles = {}
    for group, ids in joint_ids.items():
      positions = np.concatenate([root_position[:, None], joint_positions[:, ids]], axis=1)
      forward = (positions * forward_axis[:, None]).sum(-1) / forward_axis_norm_sq[:, None]
      side_profiles[group] = np.stack([forward, positions[..., 2]], axis=-1)  # [b, 1 + n, 2]
    return side_profiles

  return record
