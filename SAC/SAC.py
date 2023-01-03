# Import RL Modules
from SAC.Network import DQN, polyak_average, DEVICE
from SAC.ReplayBuffer import ReplayBuffer, RLDataset
from SAC.Policy import GradientPolicy
from SAC.Environment import create_environment
from SAC.Utils import AUTO

# Import Utilities
import copy, itertools, random
from termcolor import colored
from typing import Union, Optional
from math import pow

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# Create SAC Algorithm
class WCSAC(LightningModule):
    
    # env_name:             Environment Name
    # record_video:         Record Video with RecordVideo Wrapper
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
    
    # fixed_cost_penalty:   Cost Penalty Coefficient                    |  Set it to 'auto' to Automatically Learn
    # cost_constraint:      Adjust Cost Penalty to Maintain this Cost   |  When 'fixed_cost_penalty' is None
    # cost_limit:           Compute an Approximate Cost Constraint      |  When 'cost_constraint' is None

    def __init__(
        
        self, env_name, record_video=True, capacity=100_000, batch_size=512, hidden_size=256,
        lr=1e-3, gamma=0.99, loss_function='smooth_l1_loss', optim='AdamW', 
        epsilon=0.05, samples_per_epoch=10_000, tau=0.05,
                 
        # Entropy Coefficient α, if AUTO -> Automatic Learning:
        alpha: Union[str, float]=AUTO,
        init_alpha: Optional[float]=None,
        target_alpha: Union[str, float]=AUTO,
        
        # Cost Constraints:
        fixed_cost_penalty: Optional[float]=None,
        cost_constraint: Optional[float]=None,
        cost_limit: Optional[float]=None
        
    ):

        super().__init__()

        # Create Environment
        self.env = create_environment(env_name, record_video)
        
        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        self.max_episode_steps = self.env.spec.max_episode_steps
        
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
        
        # Create Cost NN
        self.q_net_cost = DQN(hidden_size, obs_size, action_dims).to(DEVICE)

        # Create Target Networks
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net_cost = copy.deepcopy(self.q_net_cost)
        
        # Instantiate Loss Function and Optimizer
        self.loss_function = getattr(torch.nn.functional, loss_function)
        self.optim = getattr(torch.optim, optim)
        
        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()
        
        # Initialize Entropy Coefficient (Alpha) and Constraint Coefficient (Beta)
        self.__init_coefficients(alpha, init_alpha, target_alpha, fixed_cost_penalty, cost_constraint, cost_limit)

        # Fill the ReplayBuffer
        self.__collect_experience()
        
    # Method for Filling the ReplayBuffer
    def __collect_experience(self):
        
        print(colored('\n\nStart Collecting Experience\n\n','yellow'))

        # While Buffer is Filling
        while len(self.buffer) < self.hparams.samples_per_epoch:

            if round(len(self.buffer), -2) % 500 == 0:
                print(f"{round(len(self.buffer), -2)} samples in Experience Buffer. Filling...")

            # Method that Play the Environment
            self.play_episode()

        print(colored('\n\nEnd Collecting Experience\n\n','yellow'))
    
    # Coefficient Initialization
    def __init_coefficients(
        
        self,
        
        # Entropy Coefficient α, if 'auto' -> Automatic Learning
        alpha: Union[str, float]=AUTO,
        init_alpha: Optional[float]=None,
        target_alpha: Union[str, float]=AUTO,
                  
        # Cost Constraints
        fixed_cost_penalty: Optional[float]=None,
        cost_constraint: Optional[float]=None,
        cost_limit: Optional[float]=None
    
    ):
        
        # Automatically Learn Alpha
        if alpha == AUTO:
            
            # Automatically Set Target Entropy | else: Force Float Conversion -> target α (safety-gym = -2.0)
            if target_alpha == AUTO: self.target_alpha = - torch.prod(Tensor(self.env.action_space.shape)).item()
            else: self.target_alpha = float(target_alpha)

            # FIX: Instantiate Alpha, Log_Alpha
            alpha = torch.ones(1, device=DEVICE) * (float(init_alpha) if init_alpha is not None else 0.0)
            self.log_alpha = torch.nn.Parameter(torch.log(F.softplus(alpha)), requires_grad=True)
            # alpha_ = torch.ones(1, device=DEVICE) * (float(init_alpha) if init_alpha is not None else 0.0)
            # self.alpha = torch.nn.Parameter(F.softplus(alpha_), requires_grad=True)
            # self.log_alpha = torch.log(self.alpha)
        
        # Use Cost = False is All of the Cost Arguments are None
        self.use_cost = bool(fixed_cost_penalty or cost_constraint or cost_limit)

        # Automatically Compute Cost Penalty
        if self.use_cost and fixed_cost_penalty is None:
            
            # FIX: Instantiate Beta and Log_Beta
            self.beta = torch.nn.Parameter(F.softplus(torch.zeros(1, device=DEVICE)), requires_grad=True)
            self.log_beta = torch.log(self.beta)
            # beta = torch.zeros(1, device=DEVICE)
            # self.log_beta = torch.nn.Parameter(torch.log(F.softplus(beta)), requires_grad=True)
            
            # Compute Cost Constraint
            if cost_constraint is None:
                
                ''' 
                Convert assuming equal cost accumulated each step
                Note this isn't the case, since the early in episode doesn't usually have cost,
                but since our algorithm optimizes the discounted infinite horizon from each entry
                in the replay buffer, we should be approximately correct here.
                It's worth checking empirical total undiscounted costs to see if they match. 
                '''
                
                # FIX: a ** b = pow(a,b)
                gamma, max_len = self.hparams.gamma, self.max_episode_steps
                self.cost_constraint = cost_limit * (1 - pow(gamma, max_len)) / (1 - gamma) / max_len

    @property
    # Alpha (Entropy Bonus) Computation Function
    def __alpha(self): 
        
        # FIX: Return 'log_alpha' / 'alpha' if AUTO | else: Force Float Conversion -> α
        if self.hparams.alpha == AUTO: return self.log_alpha.exp().detach()
        # if self.hparams.alpha == AUTO: return self.alpha.detach()
        else: return float(self.hparams.alpha)

    @property
    # Beta (Cost Penalty) Computation Function 
    def __beta(self):
        
        # Cost Not Contribute to Policy Optimization
        if not self.use_cost: 
            return 0.0
        
        # Use Fixed Cost Penalty
        if self.hparams.fixed_cost_penalty is not None:
            return float(self.hparams.fixed_cost_penalty)
        
        # FIX: Else Return 'log_beta' / 'beta'
        # return self.log_beta.exp().detach()
        return self.beta.detach()

    @torch.no_grad()
    def play_episode(self, policy=None):

        # Reset Environment
        obs, _ = self.env.reset()
        done = truncated = False
        episode_cost = 0.0

        while not done and not truncated:

            # Select an Action using our Policy or Random Action (in the beginning or if random < epsilon)
            if policy and random.random() > self.hparams.epsilon:
                
                # Get only the Action, not the Log Probability
                action, _ = policy(obs)
                action = action.cpu().numpy()
            
            # Sample from the Action Space
            else: action = self.env.action_space.sample()

            # Execute Action on the Environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
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

    def forward(self, x):
        
        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)
    
    def configure_optimizers(self):
        
        # Create an Iterator with the Parameters of the Critic Q-Networks and the Q-Cost
        q_net_params = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters(), self.q_net_cost.parameters())
        
        # We need 2 Separate Optimizers as we have 2 Neural Networks
        q_net_optimizer  = self.optim(q_net_params,  lr=self.hparams.lr)
        policy_optimizer = self.optim(self.policy.parameters(), lr=self.hparams.lr)
        
        # Default Optimizers
        optimizers = [q_net_optimizer, policy_optimizer]
        self.optimizers_list = ['q_net_optimizer', 'policy_optimizer']
        
        # Entropy Regularization Optimizer
        if self.hparams.alpha == AUTO:
            
            # FIX: Create Alpha Optimizer
            alpha_optimizer = self.optim([self.log_alpha], lr=self.hparams.lr)
            # alpha_optimizer = self.optim([self.alpha], lr=self.hparams.lr)

            # Append Optimizer
            optimizers.append(alpha_optimizer)
            self.optimizers_list.append('alpha_optimizer')

        # Cost Penalty Optimizer
        if self.use_cost and self.hparams.fixed_cost_penalty is None:
            
            # FIX: Create Beta Optimizer
            # cost_penalty_optimizer = self.optim([self.log_beta], lr=self.hparams.lr)
            cost_penalty_optimizer = self.optim([self.beta], lr=self.hparams.lr)

            # Append Optimizer
            optimizers.append(cost_penalty_optimizer)
            self.optimizers_list.append('cost_penalty_optimizer')

        return optimizers
    
    def train_dataloader(self):
        
        # Create a Dataset from the ReplayBuffer
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        
        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, num_workers=16)
        
        return dataloader
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        # Un-Pack the Batch Tuple
        states, actions, rewards, costs, dones, next_states = batch
        
        # Create an Extra Dimension to have Matrix in which first row stops at first element, second rows stop at second element...
        rewards = rewards.unsqueeze(1)
        costs = costs.unsqueeze(1)
        dones = dones.unsqueeze(1)
        
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
            q_loss1 = self.loss_function(action_values1, expected_action_values)
            q_loss2 = self.loss_function(action_values2, expected_action_values)
            q_cost_loss = self.loss_function(cost_values, expected_cost_values)
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
        elif 'alpha_optimizer' in self.optimizers_list and optimizer_idx == self.optimizers_list.index('alpha_optimizer'):
            
            # Compute the Actions and the Log Probabilities
            _, log_probs = self.policy(states)

            # FIX: Compute the Alpha Loss
            alpha_loss = - self.log_alpha * (torch.mean(log_probs) + self.target_alpha)
            # alpha_loss = - self.alpha * (torch.mean(log_probs) + self.target_alpha)
            self.log('episode/Alpha Loss', alpha_loss)

            return alpha_loss
        
        # Update Beta
        elif 'cost_penalty_optimizer' in self.optimizers_list and optimizer_idx == self.optimizers_list.index('cost_penalty_optimizer'):
            
            # Compute the Cost
            cost_values = self.q_net_cost(states, actions)

            # FIX: Compute the Beta Loss
            beta_loss = self.beta * (self.cost_constraint - torch.mean(cost_values))
            # beta_loss = - self.log_beta * (self.cost_constraint - torch.mean(cost_values))
            self.log('episode/Beta Loss', beta_loss)

            return beta_loss

    def training_epoch_end(self, training_step_outputs):
        
        # Play Episode
        self.play_episode(policy=self.policy)
        
        # Polyak Average Update of the Networks
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.q_net_cost, self.target_q_net_cost, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        
        # Log Episode Return
        self.log("episode/Return", self.env.return_queue[-1].item())
