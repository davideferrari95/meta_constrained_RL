from typing import Iterable, Callable

# Import Torch IterableDataset
from torch.utils.data.dataset import IterableDataset

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
