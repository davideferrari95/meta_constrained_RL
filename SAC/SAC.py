# Import RL Modules
from SAC.Network import DQN, polyak_average, DEVICE
from SAC.ReplayBuffer import ReplayBuffer, RLDataset
from SAC.Policy import GradientPolicy
from SAC.Environment import create_parallel_environment, create_test_environment
from SAC.Utils import AUTO

# Import Utilities
import copy, itertools, random
from termcolor import colored
from typing import Union, Optional

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from pytorch_lightning import LightningModule

# Create SAC Algorithm
class SAC(LightningModule):
    
    # env_name:             Environment Name
    # num_envs:             Number of Parallel Environments ~ 120MB RAM Each ('Safexp-PointGoal2-v0')
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
    
    # alpha:                Entropy Regularization Coefficient  |  Set it to 'auto' to Automatically Learn
    # init_alpha:           Initial Value of α [Optional]       |  When Learning Automatically α
    # target_alpha:         Target Entropy                      |  When Learning Automatically α, Set it to 'auto' to Automatically Compute Target    
    
    # beta:                 Cost Coefficient                    |  Set it to 'auto' to Automatically Learn
    # init_beta:            Initial Value of β [Optional]       |  When Learning Automatically β
    # target_beta:          Target Cost                         |  When Learning Automatically β, Set it to 'auto' to Automatically Compute Target

    def __init__(
        
        self, env_name, num_envs=100, capacity=1000, batch_size=512, hidden_size=256,
        lr=1e-3, gamma=0.99, loss_function=F.smooth_l1_loss, optim=AdamW, 
        epsilon=0.05, samples_per_epoch=1000, tau=0.05,
                 
        # Entropy Coefficient α, if AUTO -> Automatic Learning
        alpha: Union[str, float]=AUTO,
        init_alpha: Optional[float]=None,
        target_alpha: Union[str, float]=AUTO,
        
        # Constraint Coefficient β, if AUTO -> Automatic Learning 
        beta: Union[str, float]=AUTO,
        init_beta: Optional[float]=None,
        target_beta:Union[str, float]=AUTO
        
    ):

        super().__init__()

        # Create 'num_envs' Parallel Environments
        self.envs = create_parallel_environment(name=env_name, num_envs=num_envs)

        # Action, Observation Dimension of a Single Environment
        obs_size = self.envs.single_observation_space.shape[0]
        action_dims = self.envs.single_action_space.shape[0]

        # Get the Max Action Space Values
        max_action = self.envs.action_space.high

        # Create NN (Critic 1,2), Policy (Actor) and Replay Buffer
        self.q_net1 = DQN(hidden_size, obs_size, action_dims).to(DEVICE)
        self.q_net2 = DQN(hidden_size, obs_size, action_dims).to(DEVICE)
        self.policy = GradientPolicy(hidden_size, obs_size, action_dims, max_action).to(DEVICE)
        self.buffer = ReplayBuffer(capacity)
        
        # Create Cost NN
        self.q_net_cost = DQN(hidden_size, obs_size, action_dims).to(DEVICE)

        # Create Target Networks
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net_cost = copy.deepcopy(self.q_net_cost)

        # Initialize Entropy Coefficient (Alpha) and Constraint Coefficient (Beta)
        self.init_coefficients(alpha, init_alpha, target_alpha, beta, init_beta, target_beta)

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

    def init_coefficients(
        
        self,
        
        # Entropy Coefficient α, if 'auto' -> Automatic Learning
        alpha: Union[str, float]=AUTO,
        init_alpha: Optional[float]=None,
        target_alpha: Union[str, float]=AUTO,
                  
        # Constraint Coefficient β, if 'auto' -> Automatic Learning 
        beta: Union[str, float]=AUTO,
        init_beta: Optional[float]=None,
        target_beta: Union[str, float]=AUTO
    
    ):
        
        # Automatically Learn Alpha
        if alpha == AUTO:
            
            # Automatically Set Target Entropy | else: Force Float Conversion -> target α
            if target_alpha == AUTO: self.target_alpha = - torch.prod(Tensor(self.envs.single_action_space.shape)).item()
            else: self.target_alpha = float(target_alpha)

            # Compute Log Alpha
            self.log_alpha = Parameter(torch.log(torch.ones(1, device=DEVICE) * (float(init_alpha) if init_alpha is not None else 1.0)), requires_grad=True)

    @property
    # Alpha Computation Function
    def __alpha(self): 
        
        # Return 'log_alpha' if AUTO | else: Force Float Conversion -> α
        if self.hparams.alpha == AUTO: return self.log_alpha.exp().detach()
        else: return float(self.hparams.alpha)

    @property
    # Beta Computation Function
    def __beta(self): 
        
        # Return 'log_beta' if AUTO | else: Force Float Conversion -> β
        # if self.hparams.beta == AUTO: return self.log_beta.exp().detach()
        # else: return float(self.hparams.beta)
        return float(self.hparams.beta)

    @torch.no_grad()
    def play_episode(self, policy=None):

        # Reset Environment
        obs, _ = self.envs.reset()
        done = truncated = False
        episode_cost = 0.0

        while not done and not truncated:

            # Select an Action using our Policy or Random Action (in the beginning or if random < epsilon)
            if policy and random.random() > self.hparams.epsilon:
                
                # Get only the Action, not the Log Probability
                action, _ = policy(obs)
                action = action.cpu().numpy()
            
            # Sample from the Action Space
            else: action = self.envs.action_space.sample()

            # Execute Action on the Environment
            next_obs, reward, done, truncated, info = self.envs.step(action)
            
            # Get Cumulative Cost from the Info Dict 
            cost = info.get('cost', 0)

            # Update Episode Cost
            episode_cost += cost
            
            # Save Experience (with cost) in Replay Buffer
            exp = (obs, action, reward, cost, done, next_obs)
            self.buffer.append(exp)
        
            # Update State
            obs = next_obs
        
        # Log Episode Cost
        if policy: self.log('episode/Cost', episode_cost)

    @torch.no_grad()
    def play_test_episode(self, policy=None):
        
        # Make Test Environment
        test_env = create_test_environment(self.hparams.env_name)

        # Reset Environment
        obs, _ = test_env.reset()
        done = truncated = False

        while not done and not truncated:

            # Select an Action using our Policy or Random Sampling
            if policy: action, _ = policy(obs).cpu().numpy()
            else: action = test_env.action_space.sample()

            # Execute Action on the Environment
            next_obs, _, done, truncated, _ = test_env.step(action)
        
            # Update State
            obs = next_obs

    def forward(self, x):
        
        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)
    
    def configure_optimizers(self):
        
        # Create an Iterator with the Parameters of the Critic Q-Networks and the Q-Cost
        q_net_params = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters(), self.q_net_cost.parameters())
        
        # We need 2 Separate Optimizers as we have 2 Neural Networks
        q_net_optimizer  = self.hparams.optim(q_net_params,  lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        
        # Entropy Regularization Optimizer
        alpha_optimizer = self.hparams.optim([self.log_alpha], lr=self.hparams.lr)

        return [q_net_optimizer, policy_optimizer, alpha_optimizer]
    
    def train_dataloader(self):
        
        # Create a Dataset from the ReplayBuffer
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        
        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, num_workers=16)
        
        return dataloader
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        # Un-Pack the Batch Tuple
        states, actions, rewards, costs, dones, next_states = batch
        # states, actions, rewards, costs, dones, next_states = map(torch.squeeze, batch)
        
        # states = states.squeeze()
        print('states: ', type(states))
        print('actions: ', type(actions))
        print('rewards: ', type(rewards))
        print('costs: ', type(costs))
        print('dones: ', type(dones))
        print('next_states: ', type(next_states))
        print(states.shape)
        print(actions.shape)
        print(rewards.shape)
        print(costs.shape)
        print(dones.shape)
        print(next_states.shape)
        
        # Create an Extra Dimension to have Matrix in which first row stops at first element, second rows stop at second element...
        rewards = rewards.unsqueeze(1)
        costs = costs.unsqueeze(1)
        dones = dones.unsqueeze(1).bool()
        
        # Update Q-Networks
        if optimizer_idx == 0:
            
            # Compute the Action-Value Pair
            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)
            
            # Compute the Cost
            cost_values = self.q_net_cost(states, actions)
            
            # Compute Next Action and Log Probabilities
            target_actions, target_log_probs = self.target_policy(next_states)
            
            # Compute Next Action Values
            next_action_values = torch.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions)
            )
            
            # Compute Next Cost Values
            next_cost_values = self.target_q_net_cost(next_states, target_actions)

            # When episode is over (Done=True) we don't expect any new reward / cost -> (1-done)
            next_action_values[dones] = 0.0
            next_cost_values[dones] = 0.0
            
            # Construct the Target, Adjusting the Value with α * log(π) 
            expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.__alpha * target_log_probs)
            
            # Construct Cost Target
            expected_cost_values = costs + self.hparams.gamma * (next_cost_values)
            
            # Compute the Loss Function
            q_loss1 = self.hparams.loss_function(action_values1, expected_action_values)
            q_loss2 = self.hparams.loss_function(action_values2, expected_action_values)
            q_cost_loss = self.hparams.loss_function(cost_values, expected_cost_values)
            total_loss = q_loss1 + q_loss2 + q_cost_loss
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
            
            # Compute the Cost
            cost_values = self.q_net_cost(states, actions)

            # Compute the Policy Loss (α * Entropy + β * Cost)
            policy_loss = (self.__alpha * log_probs - action_values + self.__beta * cost_values).mean()
            self.log('episode/Policy Loss', policy_loss)

            return policy_loss
        
        # Update Alpha
        elif optimizer_idx == 2:
            
            # Compute the Actions and the Log Probabilities
            actions, log_probs = self.policy(states)

            # Use Q-Networks to Evaluate the Actions Selected by the Policy
            action_values = torch.min(
                self.q_net1(states, actions),
                self.q_net2(states, actions)
            )

            # Compute the Alpha Loss
            alpha_loss = - (self.log_alpha * (log_probs + self.target_alpha).detach()).mean()
            self.log('episode/Alpha Loss', alpha_loss)

            return alpha_loss
        

    def training_epoch_end(self, training_step_outputs):
        
        # Play Episode
        self.play_episode(policy=self.policy)
        
        # Polyak Average Update of the Networks
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.q_net_cost, self.target_q_net_cost, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        
        # Test Episode Every 50 Epochs
        if self.current_epoch % 50 == 0: 
            self.play_test_episode(policy=self.policy)
        
        # Log Episode Return
        self.log("episode/Return", self.env.return_queue[-1].item())
