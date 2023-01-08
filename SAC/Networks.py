import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np


# Select Training Device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Neural Network Creation Function
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, hidden_mod=nn.ReLU(), output_mod=None):
    
    # No Hidden Layers
    if hidden_depth == 0:
        
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
class DQN(nn.Module):
    
    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        
        # Create a Sequential Neural Network
        self.net = mlp(obs_size + out_dims, hidden_size, output_dim=1, hidden_depth=2)
        
    def forward(self, state, action):
        
        # Dimensional Check
        assert state.size(0) == action.size(0)
        
        # Convert state / action to Tensor if are numpy array
        if isinstance(state,  np.ndarray): state  = torch.from_numpy(state).to(DEVICE)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(DEVICE)
    
        # Concatenate the Features of State and Action
        obs_action = torch.cat([state, action], dim=-1)
        
        # Pass the State-Action Pair through the Network
        return self.net(obs_action.float())


# Safety Critic Network for estimating Long Term Costs (Mean and Variance)
class SafetyCritic(nn.Module):
    
    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()

        # Create the 2 Cost Networks
        self.QC = mlp(obs_size + out_dims, hidden_size, 1, 2)
        self.VC = mlp(obs_size + out_dims, hidden_size, 1, 2)

    def forward(self, state, action):

        # Dimensional Check
        assert state.size(0) == action.size(0)

        # Convert state / action to Tensor if are numpy array
        if isinstance(state,  np.ndarray): state  = torch.from_numpy(state).to(DEVICE)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(DEVICE)
        
        # Concatenates the Sequence of Tensors in the Given Dimension
        obs_action = torch.cat([state, action], dim=-1)
        
        # Pass the State-Action Pair through the Networks
        return self.QC(obs_action), self.VC(obs_action)


# Gaussian Policy Network
class GradientPolicy(nn.Module):
    
    def __init__(self, hidden_size, obs_size, out_dims, max):
        super().__init__()
        
        self.max = torch.tensor(np.array(max), device=DEVICE)
        
        # Create the Network
        self.net = nn.Sequential(
            
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Mean and Standard Deviation of the Gaussian Distribution
        self.linear_mu  = nn.Linear(hidden_size, out_dims)
        self.linear_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
    
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
        dist = Normal(mu, std)
        
        # Sample an Action from the Distribution
        action = dist.rsample()
        
        # Save the Logarithm of these Actions to use for Entropy Regularization
        log_prob = dist.log_prob(action)
        
        # Sum the Value to Obtain a Single Probability (Instead of a Vector of N Probabilities)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Probability of an Action in [0,1] -> Log in [-inf,0] -> Numerical Instability -> Trick Formula (Squashing Function)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        # Map the Output in [-1,+1] and Scale it to the Env Range
        action = torch.tanh(action) * self.max

        return action, log_prob


# Polyak Average Function to update the Target Parameters
def polyak_average(net, target_net, tau=0.01):
    
    # For every parameter of the Q-Network and Target-Network
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        
        # Polyak Average Function
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)
