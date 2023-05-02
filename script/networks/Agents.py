import torch, torch.nn as nn
from torch import distributions as TD

import gym
import numpy as np

# Select Training Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Initialization Layer Function
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    # Orthogonal Initialization on the Layer Weights
    torch.nn.init.orthogonal_(layer.weight, std)

    # Constant Initialization on the Layer Bias
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer

class PPO_Agent(nn.Module):

    def __init__(self, env:gym.Env):

        super(PPO_Agent, self).__init__()

        # Create Critic Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        self.critic = nn.Sequential(

            # Input Shape is the Product of Observation Space Shape | Tanh Activation
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)), nn.Tanh(),

            # Hidden Layer | Tanh Activation
            layer_init(nn.Linear(64, 64)), nn.Tanh(),

            # Last Layer uses 1.0 as Standard Deviation instead of the Default sqrt(2)
            layer_init(nn.Linear(64, 1), std=1.0)

        )

        # Create Actor Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        self.actor_mean = nn.Sequential(

            # Input Shape is the Product of Observation Space Shape | Tanh Activation
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)), nn.Tanh(),

            # Hidden Layer | Tanh Activation
            layer_init(nn.Linear(64, 64)), nn.Tanh(),

            # Last Layer uses 0.01 as Standard Deviation instead of the Default sqrt(2)
            # Ensures that the Layer Parameters will have Similar Scalar Values -> The Probability of Taking each Action will be Similar
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01)

        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):

        """ Pass Observations through the Critic Network """

        return self.critic(x)

    def get_action_and_value(self, x, action=None):

        """ Pass Observations through the Actor Network """

        # Un-Normalized Action Probabilities
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd)

        # Pass the Logits to a Normal Distribution
        probs = TD.Normal(action_mean, action_std)

        # Sample the Action in the Rollout Phase
        if action is None: action = probs.sample()

        # Return Actions, Log Probability, Entropy, Values
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
