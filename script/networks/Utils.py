import numpy as np
from scipy import signal
from typing import Tuple

def combined_shape(length, shape=None) -> Tuple:

    """ Combine Length and Shape of an Array """

    if shape is None: return (length,)

    # Return Combined Shape of an Array
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumulative_sum(x, discount):

    """
    Compute Discounted Cumulative Sums of Vectors.
        input: vector x, [x0, x1, x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """

    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x, with_min_and_max=False) -> Tuple:

    """
    Get mean/std and optional min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std.
    """

    # Convert to Float32 Numpy Array
    x = np.array(x, dtype=np.float32)

    # Compute Mean
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    # Compute Standard Deviation
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)

    if with_min_and_max:

        # Compute Min and Max
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf

        # Return Mean, Std, Min, Max
        return mean, std, global_min, global_max

    # Return Mean, Std
    return mean, std
