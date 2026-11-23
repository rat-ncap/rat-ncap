"""PPO restart and optimizer diagnostics for collapse.

- `DiagnosedClippedRatio` updates the actor as `ClippedRatio` and logs the first `logged_steps`
  actor steps of each update to `actor_steps.csv`, splitting each Adam step into its momentum and
  fresh-gradient parts. Candidate fixes, off by default: `reset_momentum` zeroes Adam's momentum at
  each update; `rollback` measures the KL after each step and, above `kl_threshold`, undoes the
  step, zeroes the momentum, and stops the update.
- `RestartPPO` continues PPO from a checkpoint. Checkpoints hold no Adam state, so it first rebuilds
  that state with `warmup_updates` updates at learning rate 0: a cold Adam takes
  sign-of-gradient first steps, which would be an artifact.

The provided PPO configs use the rollback fix only, and start from scratch: a run restarts from a
checkpoint only when `init_checkpoint` is set.
"""
import csv
import pathlib
import typing as T

import torch

import tonic.logging
import tonic.torch.agents as tta
import tonic.torch.models as ttm
import tonic.torch.updaters as ttu

# Actor parameter changes, one tensor per trainable variable, each shaped like its variable.
Deltas = list[torch.Tensor]


class DiagnosedClippedRatio(ttu.ClippedRatio):
  """PPO actor updater (`ClippedRatio`) with per-step diagnostics and optional stability fixes."""
  def __init__(
    self,
    *args: T.Any,
    logged_steps: int = 4,
    reset_momentum: bool = False,
    rollback: bool = False,
    **kwargs: T.Any,
  ):
    """Initializes the updater.

    Args:
      *args: Arguments of `ClippedRatio`.
      logged_steps: Number of actor steps logged at the start of each update.
      reset_momentum: Zero Adam's momentum at the start of each update.
      rollback: Undo a step whose KL to the update's rollout policy exceeds `kl_threshold`, zero
        Adam's momentum, and stop the update.
      **kwargs: Keyword arguments of `ClippedRatio` (e.g. `kl_threshold`).
    """
    super().__init__(*args, **kwargs)
    self.logged_steps = logged_steps
    self.reset_momentum = reset_momentum
    self.rollback = rollback
    self.update_index = -1
    self.update_steps = 0
    self.file: T.TextIO | None = None
    self.writer: csv.DictWriter | None = None

  def initialize(self, model: ttm.ActorCritic) -> None:
    """Initializes the updater, wrapping the optimizer's step to diagnose it."""
    super().initialize(model)
    self.actor = T.cast(torch.nn.Module, model.actor)
    self.names = [name for name, param in self.actor.named_parameters() if param.requires_grad]
    # `ClippedRatio.__call__` runs the update's actor steps itself: wrap the step to diagnose them.
    self.adam_step = self.optimizer.step
    self.optimizer.step = self.diagnosed_step

  def begin_update(self, index: int) -> None:
    """Starts update `index`: resets the step count and, with `reset_momentum`, the momentum."""
    self.update_index, self.update_steps = index, 0
    if self.reset_momentum:
      self.zero_momentum()

  def zero_momentum(self) -> None:
    """Zeroes Adam's first moment estimates."""
    for state in self.optimizer.state.values():
      state['exp_avg'].zero_()

  def __call__(
    self,
    observations: torch.Tensor | dict[str, torch.Tensor],
    actions: torch.Tensor | dict[str, torch.Tensor],
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Runs one actor step as `ClippedRatio`, then measures its KL and, with `rollback`, undoes it
    if it is too large.

    A rollback restores the parameters and zeroes Adam's first moments, but keeps the second
    moments (`exp_avg_sq`) and step counts that the undone step updated.
    """
    self.batch = observations, actions, log_probs
    before = [variable.detach().clone() for variable in self.variables]
    metrics = super().__call__(observations, actions, advantages, log_probs)

    # Measure the KL from the rollout policy to the stepped policy; roll a too large step back.
    with torch.no_grad():
      after = self.policy_with(None)
      kl_after = (log_probs - after.log_prob(actions).sum(-1)).mean()
      rolled_back = self.rollback and bool(kl_after > self.kl_threshold)
      if rolled_back:
        for variable, value in zip(self.variables, before, strict=True):
          variable.copy_(value)
        self.zero_momentum()
        metrics['stop'] = torch.as_tensor(True)
    metrics['kl_after'] = kl_after
    metrics['rolled_back'] = torch.as_tensor(float(rolled_back))
    return metrics

  def policy_with(self, deltas: Deltas | None) -> ttm.NormalActionDistribution:
    """Returns the actor's policy on the current batch, with its parameters changed by `deltas`."""
    params: dict[str, torch.Tensor] = (
      dict(zip(self.names, self.variables, strict=True)) if deltas is None else {
        name: variable + delta
        for name, variable, delta in zip(self.names, self.variables, deltas, strict=True)
      }
    )
    policy = torch.func.functional_call(self.actor, params, (self.batch[0],))
    return T.cast(ttm.NormalActionDistribution, policy)

  def split_step(self) -> tuple[Deltas, Deltas]:
    """Returns the next Adam step, `lr * m_hat / (sqrt(v_hat) + eps)`, as its momentum and gradient
    parts. Assumes plain Adam (no weight decay, AMSGrad, `maximize` or `capturable`) with one
    parameter group."""
    group = self.optimizer.param_groups[0]
    lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']
    momentum, gradient = [], []
    for variable in self.variables:
      state = self.optimizer.state.get(variable) or {}
      m = state.get('exp_avg', torch.zeros_like(variable))
      v = state.get('exp_avg_sq', torch.zeros_like(variable))
      t = int(state.get('step', 0)) + 1
      grad = T.cast(torch.Tensor, variable.grad)
      v = beta2 * v + (1 - beta2) * grad ** 2
      scale = -lr / (1 - beta1 ** t) / ((v / (1 - beta2 ** t)).sqrt() + eps)
      momentum.append(scale * beta1 * m)
      gradient.append(scale * (1 - beta1) * grad)
    return momentum, gradient

  @torch.no_grad()
  def diagnosed_step(self, *args: T.Any, **kwargs: T.Any) -> T.Any:
    """Takes Adam's step and logs diagnostics for the first `logged_steps` steps of an update."""
    self.update_steps += 1
    lr = self.optimizer.param_groups[0]['lr']
    if self.update_steps > self.logged_steps or lr == 0:
      return self.adam_step(*args, **kwargs)

    # The policy before the step. Its `.mean()` follows the action space (a dict), so use the
    # underlying Normal.
    before = self.policy_with(None)
    mean, std = before.distribution.loc, before.distribution.scale

    def kl_to(after: ttm.NormalActionDistribution) -> float:
      # Analytic KL(before || after), summed over action dimensions.
      after_std, after_mean = after.distribution.scale, after.distribution.loc
      kl = (
        torch.log(after_std / std)
        + (std ** 2 + (mean - after_mean) ** 2) / (2 * after_std ** 2)
        - 0.5
      )
      return kl.sum(-1).mean().item()

    def flat(deltas: Deltas) -> torch.Tensor:
      """Concatenates the tensors, flattened."""
      return torch.cat([delta.flatten() for delta in deltas])

    # Diagnose the step's momentum and gradient parts before taking it.
    momentum, gradient = self.split_step()
    step_momentum, step_gradient = flat(momentum), flat(gradient)
    step = step_momentum + step_gradient
    squared = (step ** 2).sort(descending=True).values
    squared_total = squared.sum()
    grads = flat([T.cast(torch.Tensor, variable.grad) for variable in self.variables])
    row = {
      'update': self.update_index,
      'step': self.update_steps,
      'lr': lr,
      'kl_momentum': kl_to(self.policy_with(momentum)),
      'kl_gradient': kl_to(self.policy_with(gradient)),
      'norm_momentum': step_momentum.norm().item(),
      'norm_gradient': step_gradient.norm().item(),
      'norm_step': step.norm().item(),
      'cos_momentum_gradient': (
        torch.dot(step_momentum, step_gradient)
        / (step_momentum.norm() * step_gradient.norm() + 1e-12)
      ).item(),
      'top1pct_share': (
        (squared[:max(len(squared) // 100, 1)].sum() / squared_total).item()
        if squared_total > 0 else 0.  # A zero step has no largest parameters.
      ),
      'largest_step_over_lr': (step.abs().max() / lr).item(),
      'moving_params': (step != 0).float().mean().item(),
      'grad_norm': grads.norm().item(),
    }

    # Take the step and diagnose its result.
    result = self.adam_step(*args, **kwargs)
    after = self.policy_with(None)
    row['kl_step'] = kl_to(after)
    row['kl_from_rollout'] = (self.batch[2] - after.log_prob(self.batch[1]).sum(-1)).mean().item()
    self.write(row)
    return result

  def write(self, row: dict[str, float]) -> None:
    """Appends a diagnostics row to `actor_steps.csv` in the log directory of tonic's logger (set
    by `tonic.Trainer`). The file is started anew by each run."""
    if self.writer is None:
      path = pathlib.Path(tonic.logging.getLogger().log_dir) / 'actor_steps.csv'
      self.file = open(path, 'w', newline='')
      self.writer = csv.DictWriter(self.file, fieldnames=list(row))
      self.writer.writeheader()
    self.writer.writerow(row)
    T.cast(T.TextIO, self.file).flush()


class RestartPPO(tta.PPO):
  """PPO that can start from a checkpoint, rebuilding Adam's state first (see the module)."""
  def __init__(
    self,
    *args: T.Any,
    init_checkpoint: str | None = None,
    warmup_updates: int = 0,
    **kwargs: T.Any,
  ):
    """Initializes the agent.

    Args:
      *args: Arguments of `PPO`.
      init_checkpoint: Model state dict to start from, or None to start from scratch.
      warmup_updates: Number of first updates at learning rate 0, which only rebuild Adam's state.
      **kwargs: Keyword arguments of `PPO`.
    """
    super().__init__(*args, **kwargs)
    self.init_checkpoint = init_checkpoint
    self.warmup_updates = warmup_updates
    self.updates = 0

  def initialize(self, observation_space: T.Any, action_space: T.Any, seed: int) -> None:
    """Initializes the agent and loads `init_checkpoint` into its model."""
    super().initialize(observation_space, action_space, seed)
    if self.init_checkpoint:
      state = torch.load(
        self.init_checkpoint,
        map_location=torch.get_default_device(),
        weights_only=False,
      )
      self.model.load_state_dict(state)

  @T.override
  def update(self) -> dict[str, T.Any]:
    """Runs a PPO update if the replay buffer is full (else returns no metrics). The first
    `warmup_updates` updates run at learning rate 0: they update Adam's moment estimates (and, as
    every update, the observation normalizer), not the parameters."""
    if not self.replay.is_full():
      return {}

    # Warm up at learning rate 0, which updates Adam's moment estimates only.
    optimizers = [self.actor_updater.optimizer, self.critic_updater.optimizer]
    warming = self.updates < self.warmup_updates
    if warming:
      rates = [optimizer.param_groups[0]['lr'] for optimizer in optimizers]
      for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = 0.

    # Update, telling a diagnosed actor updater which update it is.
    if isinstance(self.actor_updater, DiagnosedClippedRatio):
      self.actor_updater.begin_update(self.updates)
    metrics = super().update()

    if warming:
      for optimizer, rate in zip(optimizers, rates, strict=True):
        optimizer.param_groups[0]['lr'] = rate
    self.updates += 1
    return metrics
