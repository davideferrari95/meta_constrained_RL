import torch
import numpy as np
import scipy.signal as signal

from typing import Iterable, Callable

# Import Torch IterableDataset
from torch.utils.data.dataset import IterableDataset

def combined_shape(length, shape=None):

    if shape is None: return (length,)

    # Return the Combined Shape of the Data
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumulative_sum(x, discount):

    """
    Computing discounted cumulative sums of vectors.
    input: vector x, [x0,x1,x2]
    output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# PPO Buffer
class PPOBuffer():

    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lam=0.95, adv_norm=True):

        # Initialize the Buffer Elements (Observation, Action, Advantage, Reward, Return, Value, Log Probability)
        self.obs_buf = np.zeros(combined_shape(capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)

        # Initialize the Buffer Parameters (Gamma, Lambda, Pointer, Path Start Index, Max Size)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, capacity
        self.advantage_normalization = adv_norm

    def append(self, experience):

        """ Append one timestep of agent-environment interaction to the buffer. """

        # Check Buffer Space
        assert self.ptr < self.max_size, 'Buffer Overflow!'

        # Unpack the Experience Tuple
        obs, act, rew, val, logp = experience

        # Store the Experience in the Buffer
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.val_buf[self.ptr]  = val
        self.logp_buf[self.ptr] = logp

        # Increment the Buffer Pointer
        self.ptr += 1

    def finish_path(self, last_val=0):

        """
        Call this at the end of a trajectory, or when one gets cut off by an epoch ending.
        This looks back in the buffer to where the trajectory started, and uses rewards
        and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended because the agent reached a terminal
        state (died), and otherwise should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account for timesteps beyond the
        arbitrary episode horizon (or epoch cutoff).
        """

        # Take the Slice of the Buffer Representing a Trajectory (an Episode on the Environment)
        path_slice = slice(self.path_start_idx, self.ptr)

        # Get the Rewards and Values from the Buffer
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # TODO: GAE-Lambda Advantage Calculation - Else...
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumulative_sum(deltas, self.gamma * self.lam)

        # Rewards-To-Go Computation -> to be Targets for the Value Function
        self.ret_buf[path_slice] = discount_cumulative_sum(rews, self.gamma)[:-1]

        # Update the Path Start Index for a New Trajectory
        self.path_start_idx = self.ptr

    def get(self):

        """
        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropriately normalized (shifted to have mean zero and std one).
        Also, resets some pointers in the buffer.
        """

        # Buffer has to be Full Before you can get
        assert self.ptr == self.max_size, 'Buffer Not Full!'

        # Reset the Buffer Pointer and Path Start Index
        self.ptr, self.path_start_idx = 0, 0

        # Advantage Normalization Trick
        if self.advantage_normalization: self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)

        # TODO: Return the Data from the Buffer
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

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
