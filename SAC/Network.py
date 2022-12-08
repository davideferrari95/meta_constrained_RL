import torch, torch.nn as nn
import numpy as np

# Select Training Device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Create Deep Q-Learning Neural Network
class DQN(nn.Module):
    
    # hidden_size: size of the hidden layer
    # obs_size: dimension of observation of the state 
    # out_dims: dimension of the action space of the environment
    
    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        
        # Create a Sequential Neural Network
        self.net = nn.Sequential(
            
            # Input Layer with input dimension equal to observation state + output action dimension
            nn.Linear(obs_size + out_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        
        # Convert state / action to Tensor if are numpy array
        if isinstance(state,  np.ndarray): state  = torch.from_numpy(state).to(DEVICE)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action).to(DEVICE)
    
        # Concatenate the Features of State and Action Horizontally
        in_vector = torch.hstack((state, action))
        
        # Pass the State-Action Pair through the Network
        return self.net(in_vector.float())
        
# Polyak Average Function to update the Target Parameters
def polyak_average(net, target_net, tau=0.01):
    
    # For every parameter of the Q-Network and Target-Network
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        
        # Polyak Average Function
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)
        