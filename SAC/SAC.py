# Import RL Modules
from SAC.Network import DQN, polyak_average, DEVICE
from SAC.ReplayBuffer import ReplayBuffer, RLDataset
from SAC.Policy import GradientPolicy
from SAC.Environment import create_environment
from SAC.Utils import GYM_NEW

# Import Utilities
import copy, itertools, random
from termcolor import colored

# Import PyTorch
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# Create SAC Algorithm
class SAC(LightningModule):
    
    # env_name:             Environment Name
    # capacity:             ReplayBuffer Capacity
    # batch_size:           Size of the Batch
    # lr:                   Learning Rate
    # hidden_size           Size of the Hidden Layer
    # gamma:                Discount Factor
    # loss_function:        Loss Function to Compute the Loss Value
    # optim:                Optimizer to Train the NN
    # epsilon:              Epsilon = Prob. to Take a Random Action
    # samples_per_epoch:    How many Observation in a single Epoch
    # tau:                  How fast we apply the Polyak Average Update
    # alpha:                Importance of the Log Probability of the Selected Action

    def __init__(self, env_name, folder_name, capacity=100_000, batch_size=512, lr=1e-3,
                 hidden_size=256, gamma=0.99, loss_function=F.smooth_l1_loss, optim=AdamW, 
                 epsilon=0.05, samples_per_epoch=10_000, tau=0.05, alpha=0.02):

        super().__init__()

        # Create Environment
        self.env = create_environment(env_name, folder=folder_name)

        # Action, Observation Dimension 
        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        # Get the Max Action Space Values
        max_action = self.env.action_space.high

        # Create NN (Critic 1,2), Policy (Actor) and Replay Buffer
        self.q_net1 = DQN(hidden_size, obs_size, action_dims).to(DEVICE)
        self.q_net2 = DQN(hidden_size, obs_size, action_dims).to(DEVICE)
        self.policy = GradientPolicy(hidden_size, obs_size, action_dims, max_action).to(DEVICE)
        self.buffer = ReplayBuffer(capacity)

        # Create Target Networks
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()

        print(colored('\n\nStart Collecting Experience\n\n','yellow'))

        # While Buffer is Filling
        while len(self.buffer) < self.hparams.samples_per_epoch:

            if round(len(self.buffer), -2) % 500 == 0:
                print(f"{round(len(self.buffer), -2)} samples in Experience Buffer. Filling...")

            # Method that Play the Environment
            self.play_episode()

        print(colored('\n\nEnd Collecting Experience\n\n','yellow'))

    @torch.no_grad()
    def play_episode(self, policy=None):

        # NEW RELEASE: Reset Environment
        if GYM_NEW: obs, _ = self.env.reset()
        else: obs = self.env.reset()
        done = False

        while not done:

            # Select an Action using our Policy or Random Action (in the beginning or if random < epsilon)
            if policy and random.random() > self.hparams.epsilon:
                
                # Get only the Action, not the Log Probability
                action, _ = policy(obs)
                action = action.cpu().numpy()
            
            # Sample from the Action Space
            else: action = self.env.action_space.sample()

            # Execute Action on the Environment
            if GYM_NEW:
                # NEW RELEASE: gym>=0.26 -> observation, reward, termination, truncation, info = env.step()
                next_obs, reward, done, truncated, info = self.env.step(action)
                if truncated: done = truncated
            else: next_obs, reward, done, info = self.env.step(action)
            
            # Save Experience in Replay Buffer
            exp = (obs, action, reward, done, next_obs)
            self.buffer.append(exp)
        
            # Update State
            obs = next_obs
    
    def forward(self, x):
        
        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)
    
    def configure_optimizers(self):
        
        # Create an Iterator with the Parameters of the Critic Q-Networks
        q_net_params = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        
        # We need 2 Separate Optimizers as we have 2 Neural Networks
        q_net_optimizer  = self.hparams.optim(q_net_params,  lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)

        return [q_net_optimizer, policy_optimizer]
    
    def train_dataloader(self):
        
        # Create a Dataset from the ReplayBuffer
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        
        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, num_workers=16)
        
        return dataloader
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        # Un-Pack the Batch Tuple
        states, actions, rewards, dones, next_states = batch
        
        # Create an Extra Dimension to have Matrix in which first row stops at first element, second rows stop at second element...
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        
        if optimizer_idx == 0:
            
            # Compute the Action-Value Pair
            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)
            
            # Compute Next Action and Log Probabilities
            target_actions, target_log_probs = self.target_policy(next_states)
            
            # Compute Next Action Values
            next_action_values = torch.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions)
            )
            
            # When episode is over (Done=True) we don't expect any new reward
            next_action_values[dones] = 0.0
            
            # Construct the Target, Adjusting the Value with α * log(π) 
            expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)
            
            # Compute the Loss Function
            q_loss1 = self.hparams.loss_function(action_values1, expected_action_values)
            q_loss2 = self.hparams.loss_function(action_values2, expected_action_values)
            total_loss = q_loss1 + q_loss2
            self.log('episode/Q-Loss', total_loss)

            return total_loss

        # Update the Policy
        elif optimizer_idx == 1:

            # Compute the Actions and the Log Probabilities            
            actions, log_probs = self.policy(states)

            # Use Q-Networks to Evaluate the Actions Selected by the Policy
            action_values = torch.min(
                self.q_net1(states, actions),
                self.q_net2(states, actions)
            )

            # Compute the Policy Loss
            policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
            self.log('episode/Policy Loss', policy_loss)

            return policy_loss
    
    def training_epoch_end(self, training_step_outputs):
        
        # Play Episode
        self.play_episode(policy=self.policy)
        
        # Polyak Average Update of the Networks
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        
        # Log Episode Return
        if GYM_NEW: self.log("episode/Return", self.env.return_queue[-1].item())
