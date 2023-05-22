import torch, numpy as np
from scipy import signal
from typing import Tuple

# Select Training Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combined_shape(length, shape=None) -> Tuple:

    """ Combine Length and Shape of an Array """

    if shape is None: return (length,)

    # Return Combined Shape of an Array
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumulative_sum(x:torch.Tensor, discount:float):

    """
    Compute Discounted Cumulative Sums of Vectors.
        input: vector x, [x0, x1, x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """

    # Initialize Cumulative Sum
    cumulative_sum = torch.zeros_like(x)
    cumulative_sum[-1] = x[-1]

    # Compute Cumulative Sum
    for t in range(len(x) - 2, -1, -1):
        cumulative_sum[t] = x[t] + discount * cumulative_sum[t + 1]

    return cumulative_sum

def statistics_scalar(x:torch.Tensor, with_min_and_max:bool=False) -> Tuple:

    """
    Get mean/std and optional min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std.
    """

    # Compute Mean and Standard Deviation
    mean, std = torch.mean(x), torch.std(x)

    if with_min_and_max:

        # Compute Min and Max
        global_min = torch.min(x) if len(x) > 0 else torch.tensor(float('inf'))
        global_max = torch.max(x) if len(x) > 0 else torch.tensor(float('-inf'))

        # Return Mean, Std, Min, Max
        return mean, std, global_min, global_max

    # Return Mean, Std
    return mean, std
