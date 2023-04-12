# Import RL Modules
from SAC.Networks import DoubleQCritic, SafetyCritic, DiagGaussianPolicy
from SAC.Networks import polyak_average, DEVICE
from SAC.ReplayBuffer import ReplayBuffer, RLDataset
from SAC.Environment import create_environment, custom_environment_config, record_violation_episode
from SAC.Utils import CostMonitor, FOLDER, AUTO
from SAC.SafetyController import SafetyController, Odometry

# Import Utilities
import copy, itertools, sys, os
from termcolor import colored
from typing import Union, Optional
from tqdm import tqdm
import numpy as np

# Import Parameters Class
sys.path.append(FOLDER)
from config.config import EnvironmentParams

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
    # seed:                 Seeding Environment and Torch Network
    # record_video:         Record Video with RecordVideo Wrapper
    # record_epochs:        Record Video Every n Epochs
    # capacity:             ReplayBuffer Capacity
    # batch_size:           Size of the Batch
    # hidden_size           Size of the Hidden Layer
    # gamma:                Discount Factor
    # loss_function:        Loss Function to Compute the Loss Value
    # optim:                Optimizer to Train the NN
    # initial_samples:      How many Observation collected Before Training
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

    # Entropy Parameters
    # alpha:                Entropy Regularization Coefficient  |  Set it to 'auto' to Automatically Learn
    # init_alpha:           Initial Value of α [Optional]       |  When Learning Automatically α
    # target_alpha:         Target Entropy                      |  When Learning Automatically α, Set it to 'auto' to Automatically Compute Target
    # alpha_betas:          Coefficients for Computing Running Averages of Gradient
    # alpha_lr:             Learning Rate of Alpha Coefficient

    # Cost Parameters
    # fixed_cost_penalty:   Cost Penalty Coefficient                    |  Set it to 'auto' to Automatically Learn
    # init_beta:            Initial Value of β [Optional]               |  When Learning Automatically β
    # cost_limit:           Compute an Approximate Cost Constraint      |  When 'target_cost' is None
    # target_cost:          Adjust Cost Penalty to Maintain this Cost   |  When 'fixed_cost_penalty' is None
    # beta_betas:           Coefficients for Computing Running Averages of Gradient
    # beta_lr:              Learning Rate of Beta Coefficient
    # cost_lr_scale:        Scale Learning Rate of Safety Weight Optimizer
    # risk_level:           Risk Averse = 0, Risk Neutral = 1
    # damp_scale:           Damp Impact of Safety Constraint in Actor Update

    # Safe Parameters:
    unsafe_experience:      Activate or Not UnSafe Experience Saving in ReplayBuffer

    # Environment Parameters
    # env_name:             Environment Name
    # lidar_num_bins:       Number of Lidar Dots
    # lidar_max_dist:       Maximum distance for lidar sensitivity (if None, exponential distance)
    # lidar_exp_gain:       Scaling factor for distance in exponential distance lidar
    # lidar_type:           Type of Lidar Sensor | 'pseudo', 'natural', see self.obs_lidar()
    # reward_distance:      Dense reward multiplied by the distance moved to the goal
    # reward_goal:          Sparse reward for being inside the goal area
    # stuck_threshold:      Threshold Distance to Trigger Stuck Pos Changes
    # stuck_penalty:        Reward Penalty if Robot Get Stuck in Position
    # safety_threshold:     Lidar Threshold to Trigger Safe Action Control

    '''

    def __init__(

        self, seed=-1, record_video=True, record_epochs=100, capacity=100_000,
        batch_size=1024, hidden_size=256, loss_function='smooth_l1_loss', optim='AdamW',
        initial_samples=10_000,  gamma=0.99, tau=0.05, epsilon=0.1, smooth_lambda=0.01,

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

        # Safety Parameters:
        unsafe_experience=True,

        # Environment Configuration Parameters:
        environment_config:Optional[EnvironmentParams]=None

    ):

        super().__init__()
        
        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Remove Automatic Optimization (Multiple Optimizers)
        self.automatic_optimization = False

        # Create Environment
        env_name, env_config = custom_environment_config(environment_config)
        self.env = create_environment(env_name, env_config, seed, record_video, record_epochs)

        # Save Environment Config Parameters
        self.EC: Optional[EnvironmentParams] = environment_config

        # Create Violation Environment
        if self.EC.violation_environment: self.violation_env = create_environment(env_name, env_config, seed, record_video, 
                                                          environment_type='violation', env_epochs=self.EC.violation_env_epochs)

        # Create Test Environment
        if self.EC.test_environment: self.test_env = create_environment(env_name, env_config, seed, record_video, 
                                                environment_type='test', env_epochs=self.EC.test_env_epochs)

        # Initialize Safety Controller
        self.SafetyController = SafetyController(
            lidar_num_bins = self.EC.lidar_num_bins, 
            lidar_max_dist = self.EC.lidar_max_dist,
            lidar_exp_gain = self.EC.lidar_exp_gain,
            lidar_type     = self.EC.lidar_type,
            debug_print    = False)

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
            if self.hparams.target_alpha in [None, AUTO]: self.target_alpha = - torch.prod(Tensor(self.env.action_space.shape)).detach()
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

        # Initialize Episode Counter
        self.episode_counter = 0
        self.experience_episode_counter = 0

        # While Buffer is Filling
        while len(self.buffer) < self.hparams.initial_samples:

            if round(len(self.buffer), -2) % 500 == 0:
                print(f"{round(len(self.buffer), -2)} samples in Experience Buffer. Filling...")

            # Method that Play the Environment
            self.play_episode()
            self.experience_episode_counter += 1

        print(colored('\n\nEnd Collecting Experience\n\n','yellow'))

    @torch.no_grad()
    def play_episode(self, policy=None):

        # Compute Environment Seeding
        seed = np.random.randint(0,2**32)

        # Reset Environment
        obs, info = self.env.reset(seed=seed)
        done, truncated = False, False
        action_buffer = []

        # Initialize Odometry and Cost Monitor
        odometry = Odometry()
        monitor = CostMonitor()

        while not done and not truncated:

            # Select an Action using our Policy (Random Action in the Beginning)
            if policy:

                # Get only the Action, not the Log Probability
                action, _, _ = policy(obs)
                action = action.cpu().detach().numpy()

            # Sample from the Action Space
            else: action = self.env.action_space.sample()

            # Check if Policy Action is Safe
            unsafe_lidar, safe_action = self.SafetyController.check_safe_action(action, info['sorted_obs'], self.EC.safety_threshold)

            # Execute Safe Action on the Environment
            next_obs, reward, done, truncated, next_info = self.env.step(safe_action)

            # Save Executed Action in Action Buffer
            action_buffer.append(safe_action)

            # Penalize if Robot Get Stuck in Position
            odometry.update_odometry(next_info)
            if odometry.stuck_in_position(n=50, threshold=self.EC.stuck_threshold):

                monitor.robot_stuck += 1
                reward -= self.EC.stuck_penalty

            # Get Cumulative Cost | Update Episode Cost
            cost = monitor.compute_cost(next_info)

            # Add Cost to Reward
            # reward -= cost

            # Add Penalty Reward for Each Step not Getting the Goal Reward
            reward -= self.EC.penalty_step

            # Save Safe Experience in Replay Buffer
            self.buffer.append((obs, safe_action, reward, cost, float(done), next_obs))

            # Check if Safe Action != Action
            if self.hparams.unsafe_experience and np.not_equal(safe_action, action).any():

                # Save Unsafe Experience in Replay Buffer
                unsafe_exp = self.SafetyController.simulate_unsafe_action(obs, info['sorted_obs'], unsafe_lidar, action)
                for exp in unsafe_exp: self.buffer.append(exp)

                # Print Observation
                if self.SafetyController.debug_print: self.SafetyController.observation_print(safe_action, reward, done, truncated, next_info)

            # Update Observations
            obs, info = next_obs, next_info

        # Record Episode with Violation
        if monitor.get_episode_cost() != 0.0:
            record_violation_episode(self.violation_env, seed, action_buffer, self.current_epoch + self.experience_episode_counter)

        # Log Episode Cost
        if policy: self.log('Cost/Episode-Cost', monitor.get_episode_cost())
        if policy: self.log('Cost/Hazards-Violations', monitor.get_hazards_violation())
        if policy: self.log('Cost/Vases-Violations', monitor.get_vases_violation())
        if policy: self.log('Cost/Robot-Stuck', monitor.get_robot_stuck())

        # Increase Episode Counter
        self.episode_counter += 1

    @torch.no_grad()
    def play_test_episodes(self, test_constrained=False):

        for _ in tqdm(range(self.EC.test_episode_number)):

            # Reset Environment
            obs, info = self.test_env.reset(seed=np.random.randint(0,2**32))
            done, truncated = False, False

            while not done and not truncated:

                # Get the Action Mean
                action, _, _ = self.target_policy(obs, mean=True)
                action = action.cpu().detach().numpy()

                # Check if Policy Action is Safe
                if test_constrained: _, action = self.SafetyController.check_safe_action(action, info['sorted_obs'], self.EC.safety_threshold)

                # Execute Action on the Environment
                obs, _, done, truncated, info = self.test_env.step(action)

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
        dataset = RLDataset(self.buffer, self.hparams.batch_size)

        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, num_workers=os.cpu_count(), pin_memory=True)

        return dataloader

    def training_step(self, batch, batch_idx):

        # Un-Pack the Batch Tuple
        states, actions, rewards, costs, dones, next_states = batch

        # Create an Extra Dimension to have Matrix in which first row stops at first element, second rows stop at second element...
        rewards = rewards.unsqueeze(1)
        costs = costs.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Get List of Optimizers
        optimizers = self.optimizers()

        # Update Q-Networks
        if 'critic_optimizer' in self.optimizers_list:

            # Get Critic Optimizer
            critic_opt = optimizers[self.optimizers_list.index('critic_optimizer')]

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
            self.log('Critic/Q-Critic-Loss', q_critic_loss, on_epoch=True)
            self.log('Critic/Safety-Critic-Loss', safety_critic_loss, on_epoch=True)
            self.log('Critic/Total-Loss', total_loss, on_epoch=True)

            # Polyak Average Update of the Critic Networks
            if self.global_step % self.hparams.critic_update_freq == 0:
                polyak_average(self.q_critic.parameters(), self.target_q_critic.parameters(), tau=self.hparams.tau)
                polyak_average(self.safety_critic.parameters(), self.target_safety_critic.parameters(), tau=self.hparams.tau)

            # Optimizer Step
            critic_opt.zero_grad()
            self.manual_backward(total_loss)
            critic_opt.step()

        # Update the Policy
        if 'actor_optimizer' in self.optimizers_list:

            # Get Critic Optimizer
            actor_opt = optimizers[self.optimizers_list.index('actor_optimizer')]

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
            self.log('Actor/Policy-Loss', actor_loss, on_epoch=True)
            self.log('Actor/Policy-Entropy', - new_log_probs.mean(), on_epoch=True)
            self.log('Actor/Policy-Cost', torch.mean(actor_QC + self.pdf_cdf.cuda() * torch.sqrt(actor_VC)), on_epoch=True)
            self.log('Actor/Policy-Smooth_Loss', smooth_loss, on_epoch=True)

            # Polyak Average Update of the Actor Network
            if self.global_step % self.hparams.actor_update_freq == 0:
                polyak_average(self.policy.parameters(), self.target_policy.parameters(), tau=self.hparams.tau)

            # Optimizer Step
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            actor_opt.step()

        # Update Alpha
        if 'alpha_optimizer' in self.optimizers_list:

            # Get Alpha Optimizer
            alpha_opt = optimizers[self.optimizers_list.index('alpha_optimizer')]

            # Compute the Alpha Loss
            alpha_loss = torch.mean(self.log_alpha.exp() * (- new_log_probs - self.target_alpha).detach())
            self.log('Alpha-Beta/Alpha-Loss', alpha_loss)
            self.log('Alpha-Beta/Alpha-Value', self.log_alpha.exp())

            # Optimizer Step
            alpha_opt.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_opt.step()

        # Update Beta
        if 'beta_optimizer' in self.optimizers_list:

            # Get Beta Optimizer
            beta_opt = optimizers[self.optimizers_list.index('beta_optimizer')]

            # Compute the Beta Loss
            beta_loss = torch.mean(self.log_beta.exp() * (self.target_cost - CVaR).detach())
            self.log("Alpha-Beta/Beta-Loss", beta_loss)
            self.log("Alpha-Beta/Beta-Value", self.log_beta.exp())

            # Optimizer Step
            beta_opt.zero_grad()
            self.manual_backward(beta_loss)
            beta_opt.step()

    def on_train_epoch_end(self):

        # Play Episode
        self.play_episode(policy=self.target_policy)

        # Log Episode Return
        self.log("episode/Return", self.env.return_queue[-1].item(), on_epoch=True)
