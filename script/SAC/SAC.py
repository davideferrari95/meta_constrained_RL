# Import RL Modules
from SAC.Networks import DoubleQCritic, SafetyCritic, DiagGaussianPolicy
from SAC.Networks import polyak_average, DEVICE
from SAC.ReplayBuffer import ReplayBuffer, RLDataset
from SAC.Environment import create_environment, custom_environment_config
from SAC.Utils import set_seed_everywhere, AUTO

# Import Utilities
import copy, itertools
from termcolor import colored
from typing import Union, Optional
import numpy as np

# Import PyTorch
import torch
import torch.nn.functional as F
import torch.distributions as TD
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# Create SAC Algorithm
class WCSACP(LightningModule):
    
    '''
    
    # SAC Parameters
    # env_name:             Environment Name
    # seed:                 Seeding Environment and Torch Network
    # record_video:         Record Video with RecordVideo Wrapper
    # record_epochs:        Record Video Every n Epochs
    # capacity:             ReplayBuffer Capacity
    # batch_size:           Size of the Batch
    # hidden_size           Size of the Hidden Layer
    # gamma:                Discount Factor
    # loss_function:        Loss Function to Compute the Loss Value
    # optim:                Optimizer to Train the NN
    # samples_per_epoch:    How many Observation in a single Epoch
    # tau:                  How fast we apply the Polyak Average Update
    # epsilon:              Radius of Perturbation Set β(s,ε)
    # smooth_lambda:        Smoothness Regularization Coefficient

    # Critic Parameters
    # critic_lr:            Learning Rate of Critic Network
    # critic_betas:         Coefficients for Computing Running Averages of Gradient
    # critic_update_freq:   Update Frequency of Critic Network
    # critic_hidden_depth:  Number of Hidden Layers (-1) of Critic Networks
    
    # Actor Parameters
    # actor_lr:             Learning Rate of Actor Network
    # actor_betas:          Coefficients for Computing Running Averages of Gradient
    # actor_update_freq:    Update Frequency of Actor Network
    # log_std_bounds:       Constrain log_std Inside Bounds
    # actor_hidden_depth:   Number of Hidden Layers (-1) of Actor Network
    
    # Entropy Coefficient
    # alpha:                Entropy Regularization Coefficient  |  Set it to 'auto' to Automatically Learn
    # init_alpha:           Initial Value of α [Optional]       |  When Learning Automatically α
    # target_alpha:         Target Entropy                      |  When Learning Automatically α, Set it to 'auto' to Automatically Compute Target    
    # alpha_betas:          Coefficients for Computing Running Averages of Gradient
    # alpha_lr:             Learning Rate of Alpha Coefficient
    
    # Cost Constraints
    # fixed_cost_penalty:   Cost Penalty Coefficient                    |  Set it to 'auto' to Automatically Learn
    # init_beta:            Initial Value of β [Optional]               |  When Learning Automatically β
    # cost_limit:           Compute an Approximate Cost Constraint      |  When 'target_cost' is None
    # target_cost:          Adjust Cost Penalty to Maintain this Cost   |  When 'fixed_cost_penalty' is None
    # beta_betas:           Coefficients for Computing Running Averages of Gradient
    # beta_lr:              Learning Rate of Beta Coefficient
    # cost_lr_scale:        Scale Learning Rate of Safety Weight Optimizer
    # risk_level:           Risk Averse = 0, Risk Neutral = 1
    # damp_scale:           Damp Impact of Safety Constraint in Actor Update
    
    # Environment Parameters
    # lidar_num_bins:       Number of Lidar Dots
    # lidar_max_dist:       Maximum distance for lidar sensitivity (if None, exponential distance)
    # lidar_exp_gain:       Scaling factor for distance in exponential distance lidar
    # lidar_type:           Type of Lidar Sensor | 'pseudo', 'natural', see self.obs_lidar()

    '''
    
    def __init__(
        
        self, env_name, seed=-1, record_video=True, record_epochs=100, capacity=100_000,
        batch_size=512, hidden_size=256, loss_function='smooth_l1_loss', optim='AdamW', 
        samples_per_epoch=10_000,  gamma=0.99, tau=0.05, epsilon=0.1, smooth_lambda=0.01,
        
        # Critic and Actor Parameters:
        critic_lr=1e-3, critic_betas=[0.9, 0.999], critic_update_freq=2,
        actor_lr=1e-3,  actor_betas=[0.9, 0.999],  actor_update_freq=1,
        log_std_bounds=[-20, 2], critic_hidden_depth=2, actor_hidden_depth=2,
        
        # Entropy Coefficient α, if AUTO -> Automatic Learning:
        alpha: Union[str, float]=AUTO,
        init_alpha: Optional[float]=None,
        target_alpha: Union[str, float]=AUTO,
        alpha_betas=[0.9, 0.999],
        alpha_lr=1e-3,
        
        # Cost Constraints:
        fixed_cost_penalty: Optional[float]=None,
        init_beta: Optional[float]=None,
        target_cost: Optional[float]=None,
        cost_limit: Optional[float]=None,
        beta_betas=[0.9, 0.999],
        beta_lr=1e-3,
        cost_lr_scale=1,
        risk_level=0.5,
        damp_scale=10,
        
        # Environment Parameters:
        lidar_num_bins = 16,
        lidar_max_dist = None,
        lidar_exp_gain = 1.0,
        lidar_type = 'pseudo'

    ):

        super().__init__()
        
        # Setting Manual Seed
        assert seed != -1, f"Seed Must be Provided, Got Default Seed: {seed}"
        set_seed_everywhere(seed)

        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Create Environment
        config = custom_environment_config(lidar_num_bins, lidar_max_dist, lidar_type, lidar_exp_gain) if 'custom' in env_name else None
        self.env = create_environment(env_name, config, seed, record_video, record_epochs)

        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        self.max_episode_steps = self.env.spec.max_episode_steps
        assert self.max_episode_steps is not None, (f'self.env.spec.max_episode_steps = None')
        
        # Action, Observation Dimension 
        obs_size = int(self.env.observation_space.shape[0])
        action_dims = int(self.env.action_space.shape[0])

        # Get the Max Action Space Values
        action_range = [self.env.action_space.low, self.env.action_space.high]
        
        # Create Policy (Actor), Q-Critic, Safety-Critic and Replay Buffer
        self.policy = DiagGaussianPolicy(obs_size, hidden_size, action_dims, action_range, actor_hidden_depth, log_std_bounds).to(DEVICE)
        self.q_critic = DoubleQCritic(obs_size, hidden_size, action_dims, critic_hidden_depth).to(DEVICE)
        self.safety_critic = SafetyCritic(obs_size, hidden_size, action_dims, critic_hidden_depth).to(DEVICE)
        self.buffer = ReplayBuffer(capacity)
        
        # Create Target Networks
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_critic = copy.deepcopy(self.q_critic)
        self.target_safety_critic = copy.deepcopy(self.safety_critic)
        
        # Instantiate Loss Function and Optimizer
        self.loss_function = getattr(torch.nn.functional, loss_function)
        self.optim = getattr(torch.optim, optim)
        
        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()
        
        # Initialize Entropy Coefficient (Alpha) and Constraint Coefficient (Beta)
        self.__init_coefficients()

        # Fill the ReplayBuffer
        self.__collect_experience()
    
    # Coefficient Initialization
    def __init_coefficients(self):
        
        # Use Cost = False is All of the Cost Arguments are None
        self.use_cost = bool(self.hparams.fixed_cost_penalty or self.hparams.target_cost or self.hparams.cost_limit)
        
        # Learn Alpha and Learn Beta
        self.learn_alpha = True if self.hparams.alpha == AUTO else False
        self.learn_beta  = True if self.use_cost and self.hparams.fixed_cost_penalty is None else False
        
        # Automatically Learn Alpha
        if self.learn_alpha:
            
            # Automatically Set Target Entropy | else: Force Float Conversion -> target α (safety-gym = -2.0)
            if self.hparams.target_alpha in [None, AUTO]: self.target_alpha = - torch.prod(Tensor(self.env.action_space.shape)).item()
            else: self.target_alpha = float(self.hparams.target_alpha)

            # Instantiate Alpha, Log_Alpha
            alpha = float(self.hparams.init_alpha) if self.hparams.init_alpha is not None else 0.0
            self.log_alpha = torch.tensor(np.log(np.clip(alpha, 1e-8, 1e8)), device=DEVICE, requires_grad=True)
            
        # Automatically Compute Cost Penalty
        if self.learn_beta:
           
            # Instantiate Beta and Log_Beta
            beta = float(self.hparams.init_beta) if self.hparams.init_beta is not None else 0.0
            self.log_beta = torch.tensor(np.log(np.clip(beta, 1e-8, 1e8)), device=DEVICE, requires_grad=True)
           
            # Compute Cost Constraint
            if self.hparams.target_cost in [None, AUTO]:
                
                ''' 
                Convert assuming equal cost accumulated each step
                Note this isn't the case, since the early in episode doesn't usually have cost,
                but since our algorithm optimizes the discounted infinite horizon from each entry
                in the replay buffer, we should be approximately correct here.
                It's worth checking empirical total undiscounted costs to see if they match. 
                '''
                
                # Compute Target Cost
                gamma, max_len, cost_limit= self.hparams.gamma, self.max_episode_steps, self.hparams.cost_limit
                self.target_cost = cost_limit * (1 - gamma**max_len) / (1 - gamma) / max_len
        
        # Assert Risk Level in [0,1]
        assert 1 >= self.hparams.risk_level >= 0, f"risk_level Must be Between 0 and 1 (inclusive), Got: {self.hparams.risk_level}"
        
        # Compute PDF (Probability Density Function), CDF (Cumulative Distribution Function)
        normal = TD.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        
        # Pre-Compute CVaR (Conditional Value-at-Risk) for Standard Normal Distribution
        self.pdf_cdf = (normal.log_prob(normal.icdf(torch.tensor(self.hparams.risk_level))).exp() / self.hparams.risk_level).cuda()

    @property
    # Alpha (Entropy Bonus) Computation Function
    def alpha(self):
        
        # Return 'log_alpha' if AUTO | else: Force Float Conversion -> α
        if self.learn_alpha: return self.log_alpha.exp().detach()
        else: return float(self.hparams.alpha)

    @property
    # Beta (Cost Penalty) Computation Function 
    def beta(self):
        
        # Cost Not Contribute to Policy Optimization
        if not self.use_cost: 
            return 0.0
        
        # Use Fixed Cost Penalty
        if not self.learn_beta:
            return float(self.hparams.fixed_cost_penalty)
        
        # Else Return 'log_beta'
        return self.log_beta.exp().detach()

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

    @torch.no_grad()
    def play_episode(self, policy=None):

        # Reset Environment
        obs, _ = self.env.reset()
        done, truncated = False, False
        episode_cost = 0.0

        while not done and not truncated:

            # Select an Action using our Policy (Random Action in the Beginning)
            if policy:
                
                # Get only the Action, not the Log Probability
                action, _, _ = policy(obs)
                action = action.cpu().detach().numpy()
            
            # Sample from the Action Space
            else: action = self.env.action_space.sample()

            # Execute Action on the Environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Get Cumulative Cost from the Info Dict 
            cost = info.get('cost', 0)

            # Update Episode Cost
            episode_cost += cost
            
            # Save Experience (with cost) in Replay Buffer
            exp = (obs, action, reward, cost, float(done), next_obs)
            self.buffer.append(exp)
        
            # Update State
            obs = next_obs
        
        # Log Episode Cost
        if policy: self.log('episode/Cost', episode_cost, on_epoch=True)

    def forward(self, x):
        
        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)
    
    def configure_optimizers(self):
        
        # Create an Iterator with the Parameters of the Q-Critic and the Safety-Critic
        critic_params = itertools.chain(self.q_critic.parameters(), self.safety_critic.parameters())
        
        # Critic and Actor Optimizers
        critic_optimizer = self.optim(critic_params, lr=self.hparams.critic_lr, betas=self.hparams.critic_betas)
        actor_optimizer = self.optim(self.policy.parameters(), lr=self.hparams.actor_lr, betas=self.hparams.actor_betas)
        
        # Default Optimizers
        optimizers = [critic_optimizer, actor_optimizer]
        self.optimizers_list = ['critic_optimizer', 'actor_optimizer']
        
        # Entropy Regularization Optimizer
        if self.learn_alpha:
            
            # Create Alpha Optimizer
            alpha_optimizer = self.optim([self.log_alpha], lr=self.hparams.alpha_lr, betas=self.hparams.alpha_betas)

            # Append Optimizer
            optimizers.append(alpha_optimizer)
            self.optimizers_list.append('alpha_optimizer')

        # Cost Penalty Optimizer
        if self.learn_beta:
            
            # Create Beta Optimizer
            beta_optimizer = self.optim([self.log_beta], lr=self.hparams.beta_lr * self.hparams.cost_lr_scale, betas=self.hparams.beta_betas)

            # Append Optimizer
            optimizers.append(beta_optimizer)
            self.optimizers_list.append('beta_optimizer')

        return optimizers
    
    def train_dataloader(self):
        
        # Create a Dataset from the ReplayBuffer
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        
        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, num_workers=16, pin_memory=True)
        
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
            
            # Get Current Action-Value Estimates from the Double-Q Critic
            current_Q1, current_Q2 = self.q_critic(states, actions)
            
            # Get Current Cost Estimate from the Safety Critic
            current_QC, current_VC = self.safety_critic(states, actions)
            
            # Compute Next Action and Log Probabilities
            next_actions, next_log_probs, _ = self.policy(next_states, reparametrization=True)
            
            # Compute Next Action Values (* Unpack the Q1, Q2 Tuple) and Next Cost Values
            next_Q = torch.min(*self.target_q_critic(next_states, next_actions))
            next_QC, next_VC = self.target_safety_critic(next_states, next_actions)

            # Clamp VC in [1e-8, 1e+8]
            current_VC = torch.clamp(current_VC, min=1e-8, max=1e+8)
            next_VC    = torch.clamp(next_VC,    min=1e-8, max=1e+8)
            
            # Construct Q-Target
            target_Q = (rewards + self.hparams.gamma * (1 - dones) * (next_Q - self.alpha * next_log_probs)).detach()
            
            # Construct Cost Target
            target_QC = (costs + self.hparams.gamma * (1 - dones) * (next_QC)).detach()
            target_VC = (costs**2 - current_QC**2 + 2 * self.hparams.gamma * costs * next_QC
                        + self.hparams.gamma**2 * next_VC + self.hparams.gamma**2 * next_QC**2).detach()
            target_VC = torch.clamp(target_VC, min=1e-8, max=1e+8)
            
            # Compute the Loss Functions
            q_critic_loss = self.loss_function(current_Q1.double(), target_Q.double()) + self.loss_function(current_Q2.double(), target_Q.double())
            safety_critic_loss = self.loss_function(current_QC.double(), target_QC.double()) + \
                                torch.mean(current_VC + target_VC - 2 * torch.sqrt(current_VC * target_VC))
            total_loss = q_critic_loss + safety_critic_loss
            
            # Log Critic Loss Functions
            self.log('episode/Q-Critic-Loss', q_critic_loss, on_epoch=True)
            self.log('episode/Safety-Critic-Loss', safety_critic_loss, on_epoch=True)
            self.log('episode/Total-Loss', total_loss, on_epoch=True)

            # Polyak Average Update of the Critic Networks
            if self.global_step % self.hparams.critic_update_freq == 0:
                polyak_average(self.q_critic, self.target_q_critic, tau=self.hparams.tau)
                polyak_average(self.safety_critic, self.target_safety_critic, tau=self.hparams.tau)

            return total_loss

        # Update the Policy
        elif optimizer_idx == 1:

            # Compute the Updated Actions and the Log Probabilities
            new_actions, new_log_probs, dist = self.policy(states, reparametrization=True)

            # Compute Noise -> Actor Smoothing
            noise_dist = TD.multivariate_normal.MultivariateNormal(loc=torch.zeros(states.shape[1]), covariance_matrix=torch.eye(states.shape[1]))
            noise = noise_dist.rsample(torch.randn(states.shape[0]).shape)
            noise = torch.norm(noise, p=2, dim=1).view(states.shape[0], 1)
            noise = noise * self.hparams.epsilon * torch.rand(1)
            
            # Compute Smooth Loss
            _, _, smooth_dist = self.policy(states + noise.cuda())
            smooth_loss = torch.mean(0.5 * (TD.kl.kl_divergence(dist, smooth_dist) + TD.kl.kl_divergence(smooth_dist, dist)))

            # Use Critic Networks to Evaluate the New Actions Selected by the Policy
            actor_Q = torch.min(*self.q_critic(states, new_actions))
            actor_QC, actor_VC = self.safety_critic(states, new_actions)
            actor_VC = torch.clamp(actor_VC, min=1e-8, max=1e8)

            # Use Safety Critic to Evaluate the Actions Taken
            current_QC, current_VC = self.safety_critic(states, actions)
            current_VC = torch.clamp(current_VC, min=1e-8, max=1e8)

            # CVaR (Conditional Value-at-Risk) + Damp -> Impact of Safety Constraint in Policy Update
            CVaR = current_QC + self.pdf_cdf.cuda() * torch.sqrt(current_VC)
            damp = (self.hparams.damp_scale * torch.mean(self.target_cost - CVaR)) if self.hparams.fixed_cost_penalty is None else 0.0

            # Compute the Policy Loss (α * Entropy + β * Cost)
            actor_loss = torch.mean(self.alpha * new_log_probs - actor_Q \
                        + (self.beta - damp) * (actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC))) \
                        + self.hparams.smooth_lambda * smooth_loss
            
            # Log Actor Loss Functions
            self.log('episode/Policy-Loss', actor_loss, on_epoch=True)
            self.log('episode/Policy-Entropy', - new_log_probs.mean(), on_epoch=True)
            self.log('episode/Policy-Cost', torch.mean(actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC)), on_epoch=True)
            self.log('episode/Policy_Smooth_Loss', smooth_loss, on_epoch=True)

            # Save Var for Alpha, Beta Updates
            self.log_probs_Alpha = new_log_probs
            self.CVaR_Beta = CVaR
            
            # Polyak Average Update of the Actor Network
            if self.global_step % self.hparams.actor_update_freq == 0:
                polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)

            return actor_loss
        
        # Update Alpha        
        elif 'alpha_optimizer' in self.optimizers_list and optimizer_idx == self.optimizers_list.index('alpha_optimizer'):
            
            # Get the Log Probabilities
            log_probs = self.log_probs_Alpha

            # Compute the Alpha Loss
            alpha_loss = torch.mean(self.log_alpha.exp() * (- log_probs - self.target_alpha).detach())
            self.log('episode/Alpha-Loss', alpha_loss, on_epoch=True)
            self.log('episode/Alpha-Value', self.log_alpha.exp(), on_epoch=True)

            return alpha_loss
        
        # Update Beta
        elif 'beta_optimizer' in self.optimizers_list and optimizer_idx == self.optimizers_list.index('beta_optimizer'):

            # Get CVaR (Conditional Value-at-Risk)
            CVaR = self.CVaR_Beta

            # Compute the Beta Loss
            beta_loss = torch.mean(self.log_beta.exp() * (self.target_cost - CVaR).detach())
            self.log("episode/Beta-Loss", beta_loss, on_epoch=True)
            self.log("episode/Beta-Value", self.log_beta.exp(), on_epoch=True)

            return beta_loss

    def training_epoch_end(self, training_step_outputs):
        
        # Play Episode
        self.play_episode(policy=self.target_policy)
        
        # Log Episode Return
        self.log("episode/Return", self.env.return_queue[-1].item(), on_epoch=True)
