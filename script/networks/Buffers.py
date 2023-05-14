import torch, numpy as np
from typing import Iterable, Callable

# Import Torch IterableDataset
from torch.utils.data.dataset import IterableDataset

# Import Utilities
from utils.Utils import combined_shape, discount_cumulative_sum, statistics_scalar

class PPOBuffer:

    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, obs_shape, act_shape,
                 gae_gamma=0.99, gae_lambda=0.95, cost_gamma=0.99, cost_lambda=0.95):

        # Initialize Buffers
        self.observation_buffer = np.zeros(combined_shape(size, obs_shape), dtype=np.float32)
        self.action_buffer      = np.zeros(combined_shape(size, act_shape), dtype=np.float32)
        self.advantage_buffer   = np.zeros(size, dtype=np.float32)
        self.reward_buffer      = np.zeros(size, dtype=np.float32)
        self.return_buffer      = np.zeros(size, dtype=np.float32)
        self.value_buffer       = np.zeros(size, dtype=np.float32)
        self.log_probs_buffer   = np.zeros(size, dtype=np.float32)
        self.pi_mean_buffer     = np.zeros(size, dtype=np.float32)
        self.pi_std_buffer      = np.zeros(size, dtype=np.float32)

        # Initialize Cost Buffers
        self.cost_buffer            = np.zeros(size, dtype=np.float32)
        self.cost_advantage_buffer  = np.zeros(size, dtype=np.float32)
        self.cost_return_buffer     = np.zeros(size, dtype=np.float32)
        self.cost_value_buffer      = np.zeros(size, dtype=np.float32)

        # GAE-Lambda Parameters
        self.gae_gamma, self.gae_lambda = gae_gamma, gae_lambda
        self.cost_gamma, self.cost_lambda = cost_gamma, cost_lambda

        # Buffer Pointer and Maximum Size
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, distribution: torch.distributions.Normal, 
              observation, action, reward, value, cost, cost_value, log_probs):

        """ Append one timestep of agent-environment interaction to the buffer. """

        # Buffer Overflow Error
        assert self.ptr < self.max_size, f'Buffer Overflow: {self.ptr} > {self.max_size}'

        # Append Data to Buffers
        self.observation_buffer[self.ptr] = observation
        self.action_buffer[self.ptr]      = action
        self.reward_buffer[self.ptr]      = reward
        self.value_buffer[self.ptr]       = value
        self.cost_buffer[self.ptr]        = cost
        self.cost_value_buffer[self.ptr]  = cost_value
        self.log_probs_buffer[self.ptr]   = log_probs
        self.pi_mean_buffer[self.ptr]     = distribution.mean()
        self.pi_std_buffer[self.ptr]      = distribution.stddev()
        self.ptr += 1

    def finish_path(self, last_value=0, last_cost_value=0):

        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_value" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # Get a Slice of the Path representing the Last Episode
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.reward_buffer[path_slice], last_value)
        vals = np.append(self.value_buffer[path_slice], last_value)
        costs = np.append(self.cost_buffer[path_slice], last_cost_value)
        cost_vals = np.append(self.cost_value_buffer[path_slice], last_cost_value)

        # GAE-Lambda Advantage Calculation
        deltas = rews[:-1] + self.gae_gamma * vals[1:] - vals[:-1]
        self.advantage_buffer[path_slice] = discount_cumulative_sum(deltas, self.gae_gamma * self.gae_lambda)[:-1]

        # Rewards-To-Go -> Targets for the Value Function
        self.return_buffer[path_slice] = discount_cumulative_sum(rews, self.gae_gamma)[:-1]

        # GAE-Lambda Cost-Advantage Calculation
        cost_deltas = costs[:-1] + self.cost_gamma * cost_vals[1:] - cost_vals[:-1]
        self.cost_advantage_buffer[path_slice] = discount_cumulative_sum(cost_deltas, self.cost_gamma * self.cost_lambda)
        self.cost_return_buffer[path_slice] = discount_cumulative_sum(costs, self.cost_gamma)[:-1]

        # Update the Path Start Index
        self.path_start_idx = self.ptr

    def get(self):

        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        # Buffer has to be Full Before you Can Get
        assert self.ptr == self.max_size, f'Buffer Underflow: {self.ptr} < {self.max_size}'

        # Reset Buffer Pointers
        self.ptr, self.path_start_idx = 0, 0

        # Advantage Normalization Trick for Policy Gradient
        adv_mean, adv_std = statistics_scalar(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / (adv_std + 1e-8)

        # Center, but do NOT Rescale Advantages for Cost Gradient
        cost_adv_mean, _ = statistics_scalar(self.cost_advantage_buffer)
        self.cost_advantage_buffer -= cost_adv_mean

        # Create Data Dictionary
        data = dict(
            observations=self.observation_buffer, actions=self.action_buffer,
            advantages=self.advantage_buffer, cost_advantages=self.cost_advantage_buffer,
            returns=self.return_buffer, cost_returns=self.cost_return_buffer,
            pi_mean=self.pi_mean_buffer, pi_std=self.pi_std_buffer, log_probs=self.log_probs_buffer)

        # Return the Data as a Dictionary of Torch Tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class BatchBuffer():

    """
    A buffer for storing batches of trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self):

        # Initialize the Batch Buffer Elements (Observation, Action, Log Probability, Advantage, Q-Value)
        self.batch_states     = []
        self.batch_actions    = []
        self.batch_log_probs  = []
        self.batch_advantages = []
        self.batch_q_values   = []

        # Initialize the Episode Buffer Elements (Step, Rewards and Values)
        self.episode_step    = 0
        self.episode_rewards = []
        self.episode_values  = []

        # Initialize the Epoch Buffer Elements (Rewards)
        self.epoch_rewards   = []

        # Initialize Average Episode Reward and Length
        self.avg_ep_reward   = 0.0
        self.avg_ep_len      = 0.0
        self.avg_reward      = 0.0

    @property
    def train_data(self):

        return zip(self.batch_states, self.batch_actions, self.batch_log_probs,
                   self.batch_q_values, self.batch_advantages)

    def append_experience(self, experience):

        # Unpack the Experience Tuple
        state, action, log_prob, ep_reward, ep_value  = experience

        # Store the Experience in the Buffer
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_log_probs.append(log_prob)
        self.episode_rewards.append(ep_reward)
        self.episode_values.append(ep_value)

    def append_trajectory(self, trajectory):

        # Unpack the Trajectory Tuple
        reward, advantage, q_value = trajectory

        # Store the Trajectory in the Buffer
        self.epoch_rewards.append(sum(reward))
        self.batch_advantages = self.batch_advantages + advantage
        self.batch_q_values   = self.batch_q_values   + q_value

    def reset_batch(self):

        # Reset the Buffer Elements
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_advantages.clear()
        self.batch_q_values.clear()
        self.batch_log_probs.clear()
        self.epoch_rewards.clear()

    def reset_episode(self):

        self.episode_rewards = []
        self.episode_values  = []
        self.episode_step    = 0

# Basic Lightning Experience Source Iterable DataSet
class ExperienceSourceDataset(IterableDataset):

    """
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    # https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py

    def __init__(self, generate_batch: Callable):

        # Instance the Generate Batch Function
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable: #():

        # Call the Generate Batch Function
        iterator = self.generate_batch()
        return iterator
