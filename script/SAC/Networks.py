import torch, torch.nn as nn
import torch.nn.functional as F
from torch import distributions as TD

import numpy as np
import math

# Select Training Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Neural Network Creation Function
def mlp(input_dim:int, hidden_dim:int, output_dim:int, hidden_depth:int, hidden_mod=nn.ReLU(), output_mod=None):

    ''' Neural Network Creation Function '''

    # No Hidden Layers
    if hidden_depth <= 0:

        # Only one Linear Layer
        net = [nn.Linear(input_dim, output_dim)]

    else:

        # First Layer with ReLU Activation
        net = [nn.Linear(input_dim, hidden_dim), hidden_mod]

        # Add the Hidden Layers
        for i in range(hidden_depth - 1): 
            net += [nn.Linear(hidden_dim, hidden_dim), hidden_mod]

        # Add the Output Layer
        net.append(nn.Linear(hidden_dim, output_dim))

    if output_mod is not None:
        net.append(output_mod)

    # Create a Sequential Neural Network
    return nn.Sequential(*net)

# Deep Q-Learning Neural Network
class DoubleQCritic(nn.Module):

    def __init__(self, obs_size, hidden_size, action_dim, hidden_depth=2):
        super().__init__()

        # Create a Sequential Neural Network
        self.Q1 = mlp(obs_size + action_dim, hidden_size, 1, hidden_depth)
        self.Q2 = mlp(obs_size + action_dim, hidden_size, 1, hidden_depth)

        # Output Dict
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, state, action):

        # Dimensional Check
        assert state.size(0) == action.size(0)

        # Convert state / action to Tensor if are numpy array
        if isinstance(state,  np.ndarray): state  = torch.from_numpy(state).to(DEVICE)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(DEVICE)

        # Concatenate the Features of State and Action
        obs_action = torch.cat([state, action], dim=-1)

        # Pass the State-Action Pair through the Networks
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        # Add in the Output Dict
        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

# Safety Critic Network for estimating Long Term Costs (Mean and Variance)
class SafetyCritic(nn.Module):

    ''' Safety Critic Network for estimating Long Term Costs (Mean and Variance) '''    

    def __init__(self, obs_size, hidden_size, action_dim, hidden_depth=2):
        super().__init__()

        # Create the 2 Cost Networks
        self.QC = mlp(obs_size + action_dim, hidden_size, 1, hidden_depth)
        self.VC = mlp(obs_size + action_dim, hidden_size, 1, hidden_depth)

        # Output Dict
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, state, action):

        # Dimensional Check
        assert state.size(0) == action.size(0)

        # Convert state / action to Tensor if are numpy array
        if isinstance(state,  np.ndarray): state  = torch.from_numpy(state).to(DEVICE)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(DEVICE)

        # Concatenates the Sequence of Tensors in the Given Dimension
        obs_action = torch.cat([state, action], dim=-1)

        # Pass the State-Action Pair through the Networks
        qc = self.QC(obs_action)
        vc = self.VC(obs_action)

        # Add in the Output Dict
        self.outputs["qc"] = qc
        self.outputs["vc"] = vc

        return qc, vc

# Diagonal Gaussian Policy Network
class DiagGaussianPolicy(nn.Module):

    ''' Diagonal Gaussian Policy Network '''

    def __init__(self, obs_size, hidden_size, action_dim, action_range, hidden_depth=2, log_std_bounds=[-20, 2]):
        super().__init__()

        # Create the Network
        self.net = mlp(obs_size, hidden_size, 2 * action_dim, hidden_depth)

        # Get Action Range and Std Bounds
        self.action_range = action_range
        self.log_std_bounds = log_std_bounds

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, x, reparametrization=False, mean=False):

        # Convert Observation (x) to Tensor if is a numpy array
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).to(DEVICE)

        # Pass the Observation through the Network
        x = self.net(x.float())

        # Split the Output in two Tensors
        mu, log_std = x.chunk(2, dim=-1)

        # Constrain log_std in [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        # Get Standard Deviation
        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        # Create a Normal Distribution and Apply the Squashing Function
        dist = SquashedNormal(mu, std)

        # Sample Action with Reparametrization Trick
        if reparametrization: action = dist.rsample()

        else: 

            # Return Mean Action or Sample an Action from the Distribution
            action = dist.mean if mean else dist.sample()

            # Clamp Action in Range
            action = action.clamp(self.action_range[0].min(), self.action_range[1].max())
            # assert action.ndim == 2 and action.shape[0] == 1

            # Map the Output in [-1,+1] and Scale it to the Env Range
            # action = torch.tanh(action) * self.action_range[1]

        # Get Log Probability Distribution of the Taken Action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, dist

# Gradient Policy Network
class GradientPolicy(nn.Module):

    ''' Gradient Policy Network '''

    def __init__(self, obs_size, hidden_size, action_dim, max, hidden_depth=2):
        super().__init__()

        self.max = torch.tensor(np.array(max), device=DEVICE)

        # Create the Network
        self.net = mlp(obs_size, hidden_size, hidden_size, hidden_depth-1, output_mod=nn.ReLU())

        # Mean and Standard Deviation of the Gaussian Distribution
        self.linear_mu  = nn.Linear(hidden_size, action_dim)
        self.linear_std = nn.Linear(hidden_size, action_dim)

    def forward(self, x, reparametrization=False, mean=False):

        # Convert Observation (x) to Tensor if is a numpy array
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).to(DEVICE)

        # Pass the Observation into the Common Layer
        x = self.net(x.float())

        # Compute the Gaussian Probability Distribution
        mu  = self.linear_mu(x)
        std = self.linear_std(x)

        # Standard Deviation Must be Positive and Not Too Close to 0 (+1e-3) -> SoftPlus
        std = F.softplus(std) + 1e-3

        # Create the Gaussian Distribution
        dist = TD.normal.Normal(mu, std)

        # Sample an Action from the Distribution
        action = dist.rsample() if reparametrization else dist.sample()

        # Save the Logarithm of these Actions to use for Entropy Regularization
        log_prob = dist.log_prob(action)

        # Sum the Value to Obtain a Single Probability (Instead of a Vector of N Probabilities)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Probability of an Action in [0,1] -> Log in [-inf,0] -> Numerical Instability -> Trick Formula (Squashing Function)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        # Map the Output in [-1,+1] and Scale it to the Env Range
        action = torch.tanh(action) * self.max

        return action, log_prob

# Hyperbolic Tangent Torch Transformation
class TanhTransform(TD.transforms.Transform):

    ''' Hyperbolic Tangent Torch Transformation '''

    # Transformation Properties
    domain = TD.constraints.real
    codomain = TD.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

# Create a Gaussian Distribution and Apply the Squashing Function
class SquashedNormal(TD.transformed_distribution.TransformedDistribution):

    ''' Create a Gaussian Distribution and Apply the Squashing Function '''

    def __init__(self, mean, std):

        # Get Gaussian Mean
        self.mu = mean

        # Create a Gaussian Distribution
        self.base_dist = TD.Normal(mean, std)

        # Apply the Hyperbolic Tangent Transformation
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):

        # Get Mean
        mu = self.mu

        # Apply Transformation to Mean Value
        for tr in self.transforms: mu = tr(mu)

        return mu

# Polyak Average Function to update the Target Parameters
def polyak_average(net, target_net, tau=0.01):

    # For every parameter of the Q-Network and Target-Network
    for param, target_param in zip(net.parameters(), target_net.parameters()):

        # Polyak Average Function
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Custom Weight Init for Conv2D and Linear layers
def weight_init(m):

    ''' Custom Weight Init for Conv2D and Linear layers '''

    if isinstance(m, nn.Linear):

        # Fills the Input Tensor with a (semi) Orthogonal Matrix
        nn.init.orthogonal_(m.weight.data)

        if hasattr(m.bias, "data"): m.bias.data.fill_(0.0)
