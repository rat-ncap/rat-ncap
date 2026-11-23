"""Script used to play with trained agents."""

import argparse
import os
import typing as T

import numpy as np
import torch
import yaml

import tonic
from tonic import core
from tonic.torch import agent


def _unbatch_action(actions: core.Action) -> core.Action:
  """Extracts the first element from a batch of actions.
  
  Handles both tensor and dict action types.
  """
  if isinstance(actions, dict):
    return {k: v[0] for k, v in actions.items()}
  else:
    return actions[0]


def _batch_observation(observation: core.Observation) -> core.Observation:
  """Adds a batch dimension to an observation.
  
  Uses core.stack_tensors to handle both tensor and dict observation types.
  """
  return core.stack_tensors([observation])


class Player:
  """Player used to evaluate a trained agent in an environment."""

  def __init__(
    self,
    seed: int = 0,
  ) -> None:
    self.seed = seed
    self.agent: agent.Agent | None = None
    self.environment: core.Env | None = None

  def initialize(
    self,
    agent_instance: agent.Agent,
    environment_instance: core.Env,
  ) -> None:
    """Initializes the player with an agent and environment."""
    self.agent = agent_instance
    self.environment = environment_instance
    
    observation_space = self.environment.observation_space
    action_space = self.environment.action_space
    assert isinstance(observation_space, core.ObservationSpace)
    assert isinstance(action_space, core.ActionSpace)
    self.agent.initialize(observation_space, action_space, self.seed)

  def play(self) -> None:
    """Runs the main play loop."""
    assert self.agent is not None
    assert self.environment is not None
    
    # Determine environment type and launch appropriate player.
    environment_type: str = type(self.environment).__name__
    
    # Check if environment is wrapped in an ActionRescaler.
    if environment_type == 'ActionRescaler':
      # Get the inner environment type
      inner_env = self.environment.env  # type: ignore
      environment_type = type(inner_env).__name__

    if environment_type == 'ControlSuiteEnvironment':
      self._play_control_suite()
    else:
      if 'Bullet' in environment_type:
        self.environment.render()
      self._play_gym()

  def _play_gym(self) -> None:
    """Launches an agent in a Gym-based environment."""
    env = self.environment
    assert self.agent is not None
    assert env is not None
    
    environment = tonic.environments.distribute(lambda: env)
    environment.initialize(self.seed)

    observations, _ = environment.reset()
    environment.render()

    score = 0.0
    length = 0
    min_reward = float('inf')
    max_reward = -float('inf')
    global_min_reward = float('inf')
    global_max_reward = -float('inf')
    episodes: int = 0

    while True:
      actions = self.agent.test_step(observations)
      observations, rewards, terminations, truncations, infos = environment.step(actions)
      environment.render()

      reward = rewards[0]
      score += reward
      min_reward = min(min_reward, reward)
      max_reward = max(max_reward, reward)
      global_min_reward = min(global_min_reward, reward)
      global_max_reward = max(global_max_reward, reward)
      length += 1

      if terminations[0] or truncations[0]:
        episodes += 1

        print()
        print(f'Episodes: {episodes:,}')
        print(f'Score: {score:,.3f}')
        print(f'Length: {length:,}')
        print(f'Terminal: {terminations[0]}')
        print(f'Min reward: {min_reward:,.3f}')
        print(f'Max reward: {max_reward:,.3f}')
        print(f'Global min reward: {global_min_reward:,.3f}')
        print(f'Global max reward: {global_max_reward:,.3f}')

        score = 0.0
        length = 0
        min_reward = float('inf')
        max_reward = -float('inf')

  def _play_control_suite(self) -> None:
    """Launches an agent in a DeepMind Control Suite-based environment."""
    from dm_control import viewer

    assert self.agent is not None
    assert self.environment is not None

    class ControlSuiteWrapper:
      """Wrapper used to plug a Tonic environment in a dm_control viewer."""

      def __init__(self, env: core.Env, player_agent: agent.Agent, seed: int) -> None:
        self.environment = env
        self.agent = player_agent
        self.seed = seed
        self.unwrapped = env.unwrapped
        self.action_spec = self.unwrapped.environment.action_spec
        self.physics = self.unwrapped.environment.physics
        self.infos: dict[str, core.Observation | np.ndarray] | None = None
        self.steps = 0
        self.episodes = 0
        self.min_reward = float('inf')
        self.max_reward = -float('inf')
        self.global_min_reward = float('inf')
        self.global_max_reward = -float('inf')

      def reset(self) -> T.Any:
        """Mimics a dm_control reset for the viewer."""
        observation, _ = self.environment.reset(seed=self.seed)
        self.observations: core.Observation = _batch_observation(observation)

        self.score = 0.0
        self.length = 0
        self.min_reward = float('inf')
        self.max_reward = -float('inf')

        return self.unwrapped.last_time_step

      def step(self, actions: core.Action) -> T.Any:
        """Mimics a dm_control step for the viewer."""
        # Handle both tensor and dict actions.
        if isinstance(actions, dict):
          assert not any(np.isnan(v.sum()) for v in actions.values() if isinstance(v, torch.Tensor))
        else:
          assert not np.isnan(actions.sum())
        single_action = _unbatch_action(actions)
        
        observation, reward, terminated, truncated, info = self.environment.step(single_action)
        reward = float(reward)

        self.score += reward
        self.length += 1
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.global_min_reward = min(self.global_min_reward, reward)
        self.global_max_reward = max(self.global_max_reward, reward)
        done: bool = terminated or truncated

        if done:
          self.episodes += 1
          print()
          print(f'Episodes: {self.episodes:,}')
          print(f'Score: {self.score:,.3f}')
          print(f'Length: {self.length:,}')
          print(f'Terminal: {terminated}')
          print(f'Min reward: {self.min_reward:,.3f}')
          print(f'Max reward: {self.max_reward:,.3f}')
          print(f'Global min reward: {self.global_min_reward:,.3f}')
          print(f'Global max reward: {self.global_max_reward:,.3f}')

        self.observations = _batch_observation(observation)
        self.infos = dict(
          observations=_batch_observation(observation),
          rewards=np.array([reward]),
          truncations=np.array([truncated]),
          terminations=np.array([terminated]),
        )

        return self.unwrapped.last_time_step

    # Wrap the environment for the viewer.
    wrapper = ControlSuiteWrapper(self.environment, self.agent, self.seed)

    def policy(timestep: T.Any) -> core.Action:
      """Policy function for the dm_control viewer."""
      if wrapper.infos is not None:
        assert wrapper.agent is not None
        wrapper.steps += 1
      return wrapper.agent.test_step(wrapper.observations)

    # Launch the viewer with the wrapped environment and policy.
    viewer.launch(wrapper, policy)


def load_experiment(
  path: str | None,
  checkpoint: str,
  seed: int,
  header: str | None,
  agent_class: str | None,
  environment_class: str | None,
) -> tuple[agent.Agent, core.Env, str | None]:
  """Reloads an agent and an environment from a previous experiment."""
  checkpoint_path: str | None = None

  if path:
    print(f'Loading experiment from {path}')

    # Use no checkpoint, the agent is freshly created.
    if checkpoint == 'none' or agent_class is not None:
      print('Not loading any weights')
    else:
      checkpoint_path = os.path.join(path, 'checkpoints')
      if not os.path.isdir(checkpoint_path):
        print(f'Error: {checkpoint_path} is not a directory')
        checkpoint_path = None
      else:
        # List all the checkpoints.
        checkpoint_ids: list[int] = []
        for file in os.listdir(checkpoint_path):
          if file.startswith('step_'):
            checkpoint_id = file.split('.')[0]
            checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
          # Use the last checkpoint.
          if checkpoint == 'last':
            checkpoint_id = max(checkpoint_ids)
            checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
          # Use the specified checkpoint.
          else:
            try:
              checkpoint_id = int(checkpoint)
              if checkpoint_id in checkpoint_ids:
                checkpoint_path = os.path.join(checkpoint_path, f'step_{checkpoint_id}')
              else:
                print(f'Error: Checkpoint {checkpoint_id} not found in {checkpoint_path}')
                checkpoint_path = None
            except ValueError:
              print(f'Error: Invalid checkpoint id: {checkpoint}')
              checkpoint_path = None
        else:
          print(f'Error: No checkpoint found in {checkpoint_path}')
          checkpoint_path = None

    # Load the experiment configuration.
    if path:
      arguments_path = os.path.join(path, 'config.yaml')
      with open(arguments_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
      config = argparse.Namespace(**config)

      header = header or config.header
      agent_class = agent_class or config.agent
      environment_class = environment_class or config.test_environment
      environment_class = environment_class or config.environment

  # Run the header first, e.g. to load an ML framework.
  if header:
    exec(header)

  # Build the agent.
  if not agent_class:
    raise ValueError('No agent specified.')
  agent_instance: agent.Agent = eval(agent_class)

  # Build the environment.
  if not environment_class:
    raise ValueError('No environment specified.')
  environment_instance: core.Env = eval(environment_class)

  return agent_instance, environment_instance, checkpoint_path


def play_experiment(
  path: str | None = None,
  checkpoint: str = 'last',
  seed: int = 0,
  header: str | None = None,
  agent: str | None = None,
  environment: str | None = None,
) -> None:
  """Loads and plays an experiment with a trained agent."""
  agent_instance, environment_instance, checkpoint_path = load_experiment(path, checkpoint, seed, header, agent, environment)
  
  # Create and initialize player.
  player = Player(seed=seed)
  player.initialize(agent_instance, environment_instance)
  
  # Load the weights of the agent from a checkpoint.
  if checkpoint_path:
    agent_instance.load(checkpoint_path)
  
  # Play with the agent in the environment.
  player.play()


if __name__ == '__main__':
  # Argument parsing.
  parser = argparse.ArgumentParser(description='Play with a trained agent.')
  parser.add_argument('--path', type=str, default=None, help='Path to the experiment directory.')
  parser.add_argument('--checkpoint', type=str, default='last', help='Checkpoint to load (last, none, or checkpoint number).')
  parser.add_argument('--seed', type=int, default=0, help='Random seed.')
  parser.add_argument('--header', type=str, default=None, help='Header code to execute before loading the agent.')
  parser.add_argument('--agent', type=str, default=None, help='Agent class to instantiate.')
  parser.add_argument('--environment', '--env', type=str, default=None, help='Environment class to instantiate.')
  
  args = parser.parse_args()
  play_experiment(
    path=args.path,
    checkpoint=args.checkpoint,
    seed=args.seed,
    header=args.header,
    agent=args.agent,
    environment=args.environment,
  )

