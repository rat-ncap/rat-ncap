import torch


def lambda_returns(
  values: torch.Tensor,
  next_values: torch.Tensor,
  rewards: torch.Tensor,
  terminations: torch.Tensor,
  truncations: torch.Tensor,
  discount_factor: float,
  trace_decay: float,
) -> torch.Tensor:
  """Calculate lambda-returns per-timestep using GAE (Generalized Advantage Estimation).
  
  Computes n-step returns with eligibility traces, blending TD(0) and Monte Carlo
  estimates. The trace_decay parameter (lambda) controls the bias-variance tradeoff.
  
  Args:
    values: Value estimates for current states, shape [num_steps, num_envs].
    next_values: Value estimates for next states, shape [num_steps, num_envs].
    rewards: Rewards received, shape [num_steps, num_envs].
    terminations: Episode termination flags (true if done, false otherwise), shape [num_steps, num_envs].
    truncations: Episode truncation flags (true if reset, false otherwise), shape [num_steps, num_envs].
    discount_factor: Discount factor (gamma) for future rewards.
    trace_decay: Eligibility trace decay (lambda) for return computation.
    
  Returns:
    Lambda-returns for each timestep, shape [num_steps, num_envs].
  """
  returns = torch.zeros_like(next_values)  # [num_steps, num_envs]
  last_returns = next_values[-1]  # [num_envs]
  
  for t in reversed(range(len(rewards))):
    # Blend TD(0) bootstrap with previous lambda-return.
    bootstrap = (1 - trace_decay) * next_values[t] + trace_decay * last_returns
    # Reset bootstrap to current value on episode truncations (timeouts).
    bootstrap *= ~truncations[t]
    bootstrap += truncations[t] * next_values[t]
    # Zero out bootstrap on true terminations (no future value).
    bootstrap *= ~terminations[t]
    # Compute return: r_t + gamma * bootstrap.
    returns[t] = last_returns = rewards[t] + discount_factor * bootstrap
  
  return returns


def episodic_returns(
  values: torch.Tensor,
  next_values: torch.Tensor,
  rewards: torch.Tensor,
  terminations: torch.Tensor,
  truncations: torch.Tensor,
  discount_factor: float,
) -> torch.Tensor:
  """Calculate Monte Carlo returns averaged across episodes (possibly partial episodes).

  Computes discounted infinite-horizon returns, only bootstrapping on `truncated` timesteps, including the final timestep.

  Args:
    values: Value estimates for current states, shape [num_steps, num_envs].
    next_values: Value estimates for next states, shape [num_steps, num_envs].
    rewards: Rewards received, shape [num_steps, num_envs].
    terminations: Episode termination flags (true if done, false otherwise), shape [num_steps, num_envs].
    truncations: Episode truncation flags (true if reset, false otherwise), shape [num_steps, num_envs].
    discount_factor: Discount factor (gamma) for future rewards.

  Returns:
    Discounted returns averaged across episodes, shape [num_envs].
  """
  returns = torch.zeros_like(next_values[-1])  # [num_envs]
  last_returns = ~terminations[-1] * next_values[-1]  # [num_envs]
  resets = terminations | truncations  # [num_steps, num_envs]

  num_steps = resets.shape[0]
  for t in reversed(range(num_steps)):
    # Bootstrap return on truncations (but not terminations) on last step of episode.
    last_returns = ~resets[t] * last_returns + resets[t] * ~terminations[t] * next_values[t]
    # Add reward to current episode.
    last_returns = rewards[t] + discount_factor * last_returns
    if t == 0:
      # Add return to total for the first episode.
      returns += last_returns
    else:
      # Add return to total before starting a new episode.
      returns += resets[t-1] * last_returns

  # Average return by the number of episodes. The last timestep should always be a truncation (and 
  # possibly also a termination), so we count it manually to guarantee no division by zero.
  episodes = resets[:-1].sum(0) + 1  # [num_envs]
  returns /= episodes

  return returns
  


def flatten_batch(batch: torch.Tensor) -> torch.Tensor:
  """Flatten time and environment dimensions into a single batch dimension.
  
  Converts buffers from [num_steps, num_envs, ...] to [num_steps * num_envs, ...]
  for batched training.
  
  Args:
    batch: Tensor with shape [num_steps, num_envs, ...].
    
  Returns:
    Flattened tensor with shape [num_steps * num_envs, ...].
  """
  shape = batch.shape  # [num_steps, num_envs, ...]
  new_shape = (shape[0] * shape[1],) + shape[2:]  # [num_steps * num_envs, ...]
  return batch.reshape(new_shape)
