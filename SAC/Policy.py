import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np

from SAC.Network import DEVICE

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
        
        # Mean and Standare Deviation of the Gaussian Distribution
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
        log_prob -= (2 * (np.log(2) - action- F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        # Map the Output in [-1,+1] and Scale it to the Env Range
        action = torch.tanh(action) * self.max

        return action, log_prob
