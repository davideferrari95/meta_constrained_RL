import gym, time, numpy as np
from collections import deque
from typing import Optional

# Import Utilities Functions
from gym.wrappers.record_episode_statistics import add_vector_episode_statistics

class RecordEpisodeStatistics(gym.Wrapper):

    """ This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     ...
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since instantiation of wrapper>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since instantiation of wrapper>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes

    """

    def __init__(self, env: gym.Env, deque_size: int = 100):

        """ This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`

        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):

        """ Resets the environment using kwargs and resets the episode returns and lengths. """

        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action, ratio:int=1, simulate_in_adamba:bool=False):

        """ Steps through the environment, recording the episode statistics. """

        # Call Child `env.step` Function
        (observations, rewards, terminateds, truncateds, infos,) = self.env.step(action, ratio, simulate_in_adamba)

        assert isinstance(infos, dict), \
        f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        if not simulate_in_adamba:

            self.episode_returns += rewards
            self.episode_lengths += 1
            if not self.is_vector_env:
                terminateds = [terminateds]
                truncateds = [truncateds]
            terminateds = list(terminateds)
            truncateds = list(truncateds)

            for i in range(len(terminateds)):
                if terminateds[i] or truncateds[i]:
                    episode_return = self.episode_returns[i]
                    episode_length = self.episode_lengths[i]
                    episode_info = {
                        "episode": {
                            "r": episode_return,
                            "l": episode_length,
                            "t": round(time.perf_counter() - self.t0, 6),
                        }
                    }
                    if self.is_vector_env:
                        infos = add_vector_episode_statistics(
                            infos, episode_info["episode"], self.num_envs, i
                        )
                    else:
                        infos = {**infos, **episode_info}
                    self.return_queue.append(episode_return)
                    self.length_queue.append(episode_length)
                    self.episode_count += 1
                    self.episode_returns[i] = 0
                    self.episode_lengths[i] = 0

            return (
                observations,
                rewards,
                terminateds if self.is_vector_env else terminateds[0],
                truncateds if self.is_vector_env else truncateds[0],
                infos,
            )

        else:

            return observations, rewards, terminateds, truncateds, infos
