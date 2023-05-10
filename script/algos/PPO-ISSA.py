# Import RL Modules
from networks.Agents import PPO_Agent, DEVICE
from networks.Buffers import ExperienceSourceDataset, BatchBuffer
from envs.Environment import create_environment, record_violation_episode
from envs.DefaultEnvironment import custom_environment_config
from utils.Utils import CostMonitor, FOLDER, AUTO

# Import Utilities
import sys, gym
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np

# Import Parameters Class
sys.path.append(FOLDER)
from config.config import EnvironmentParams

# Import PyTorch
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# Create PPO Algorithm
class PPO_ISSA(LightningModule):

    def __init__(

        self,

        # Training Parameters:
        max_epochs:         int = 1000,                         # Maximum Number of Epochs
        early_stop_metric:  str = 'episode/avg_ep_reward',      # Metric for Early Stopping

        steps_per_epoch:    int = 2048,                         # How Action-State Pairs to Rollout for Trajectory Collection per Epoch
        batch_size:         int = 512,                          # Batch Size for Training
        num_mini_batches:   int = 32,                           # Number of Mini-Batches for Training
        hidden_sizes:       Optional[List[int]] = [128,128],    # Hidden Layer Sizes for Actor and Critic Networks
        hidden_mod:         Optional[str]       = 'Tanh',       # Hidden Layer Activation Function for Actor and Critic Networks

        # Optimization Parameters:
        optim:              str = 'Adam',                       # Optimizer for Critic and Actor Networks
        optim_update:       int = 4,                            # Number of Gradient Descent to Perform on Each Batch        
        lr_actor:           float = 3e-4,                       # Learning Rate for Actor Network
        lr_critic:          float = 1e-3,                       # Learning Rate for Critic Network

        # GAE (General Advantage Estimation) Parameters:
        gae_gamma:          float = 0.99,                       # Discount Factor for GAE
        gae_lambda:         float = 0.95,                       # Advantage Discount Factor (Lambda) for GAE
        adv_normalize:      bool  = True,                       # Normalize Advantage Function

        # PPO (Proximal Policy Optimization) Parameters:
        anneal_lr:          bool  = True,                       # Anneal Learning Rate
        epsilon:            float = 1e-5,                       # Epsilon for Annealing Learning Rate
        clip_ratio:         float = 0.2,                        # Clipping Parameter for PPO
        clip_gradient:      bool  = True,                       # Clip Gradient SGD
        clip_vloss:         bool  = True,                       # Clip Value Loss
        vloss_coef:         float = 0.5,                        # Value Loss Coefficient
        entropy_coef:       float = 0.01,                       # Entropy Coefficient
        max_grad_norm:      float = 0.5,                        # Maximum Gradient Norm
        target_kl:          float = 0.015,                      # Target KL Divergence

        # Environment Configuration Parameters:
        seed:               int  = -1,                          # Random Seed for Environment, Torch and Numpy
        record_video:       bool = True,                        # Record Video of the Environment
        record_epochs:      int  = 100,                         # Record Video Every N Epochs
        environment_config: Optional[EnvironmentParams] = None  # Environment Configuration Parameters

    ):

        super().__init__()

        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Remove Automatic Optimization (Multiple Optimizers)
        self.automatic_optimization = False

        # Configure Environment
        self.configure_environment(environment_config, seed, record_video, record_epochs)

        # Create PPO Agent (Policy and Value Networks)
        self.agent = PPO_Agent(self.env, hidden_sizes, getattr(torch.nn, hidden_mod)).to(DEVICE)

        # Create the Batch Buffer
        self.buffer = BatchBuffer()

        # Initialize Observation State
        state, _ = self.env.reset()
        self.state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()

        # Compute Mini-Batch Size and Add it to Hyperparameters
        self.hparams.mini_batch_size = batch_size // num_mini_batches

    def configure_environment(self, environment_config, seed, record_video, record_epochs):

        """ Configure Environment """

        # Create Environment
        env_name, env_config = custom_environment_config(environment_config)
        self.env = create_environment(env_name, env_config, seed, record_video, record_epochs)

        # Assert that the Action Space is Continuous
        assert isinstance(self.env.action_space, gym.spaces.Box), 'Only Continuous (Box) Action Space is Supported'

        # Save Environment Config Parameters
        self.EC: Optional[EnvironmentParams] = environment_config

        # Create Violation Environment
        if self.EC.violation_environment: self.violation_env = create_environment(env_name, env_config, seed, record_video, 
                                                          environment_type='violation', env_epochs=self.EC.violation_env_epochs)

        # Create Test Environment
        if self.EC.test_environment: self.test_env = create_environment(env_name, env_config, seed, record_video, 
                                                environment_type='test', env_epochs=self.EC.test_env_epochs)

        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        self.max_episode_steps = self.env.spec.max_episode_steps
        assert self.max_episode_steps is not None, (f'self.env.spec.max_episode_steps = None')

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: #():

        """ Forward Pass of the PPO Agent """

        # Pass Observation State through Actor and Critic Networks
        _, actions, log_probs, value = self.agent(x)

        return actions, log_probs, value

    def anneal_lr(self, optimizer, initial_lr):

        """ Anneal the Learning Rate """

        # Fraction Variable that Linearly Decrease to 0
        frac = 1.0 - (self.current_epoch - 1.0) / self.hparams.max_epochs
        lr_now = frac * initial_lr

        # Update the Optimizer Learning Rate
        optimizer.param_groups[0]['lr'] = lr_now

    def discount_rewards(self, rewards: List[float], discount:float) -> List[float]: #():

        """ Compute Discounted Rewards of all Rewards in the List
        Args:
            rewards: List of Rewards/Advantages
        Returns:
            List of Discounted Rewards/Advantages
        """

        # Check if Rewards are Floats
        assert isinstance(rewards[0], float), 'Rewards Items must be Floats'

        # Initialize Discounted Rewards
        cumulative_rewards, sum_r = [], 0.0

        for reward in reversed(rewards):

            # Compute Discounted Reward
            sum_r = reward + discount * sum_r
            cumulative_rewards.append(sum_r)

        return list(reversed(cumulative_rewards))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]: #():

        """Calculate the Advantage given Rewards, State Values, and the Last Value of the Episode
        Args:
            rewards:    List of Episode Rewards
            values:     List of State Values from Critic
            last_value: Value of Last State of Episode
        Returns:
            List of Advantages
        """

        # Add Last Value to Rewards and Values Lists
        rews = rewards + [last_value]
        vals = values + [last_value]

        # GAE (Generalized Advantage Estimation) Computation
        delta = [rews[i] + self.hparams.gae_gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.hparams.gae_gamma * self.hparams.gae_lambda)

        return adv

    def forward(self, x):

        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)

    @torch.no_grad()
    def play_test_episodes(self):

        for _ in tqdm(range(self.EC.test_episode_number)):

            # Reset Environment
            obs, _ = self.test_env.reset(seed=np.random.randint(0,2**32))
            done, truncated = False, False

            while not done and not truncated:

                # Get the Action Mean
                action, _, _ = self.agent.actor(obs)

                # Execute Action on the Environment
                obs, _, done, truncated, _ = self.test_env.step(action.cpu().detach().numpy())

    def generate_trajectory_samples(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: #():

        """
        Generate Trajectory Data to Train Policy and Value Networks
        Yield Tuple of Lists Containing Tensors for: States, Actions, Log Probabilities, Q-Values, Advantage
        """

        # For a Fixed Number of Steps per Epoch
        for step in range(self.hparams.steps_per_epoch):

            # Get Action and Log Probability from Policy Network
            _, action, log_prob, value = self.agent(self.state)
            next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy())

            # Store Experience Tuple in Batch Buffer
            self.buffer.append_experience((self.state, action, log_prob, reward, value.item()))

            # Update State and Episode Step
            self.buffer.episode_step += 1
            self.state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE)

            # Check Epoch End or Episode End
            epoch_end = step == (self.hparams.steps_per_epoch - 1)
            terminal = len(self.buffer.episode_rewards) == self.max_episode_steps

            # Check if Epoch End or Episode is Done or Truncated 
            if epoch_end or done or truncated or terminal:

                # Trajectory Ends Abruptly -> Bootstrap Value of Next State
                if (truncated or terminal or epoch_end) and not done:

                    with torch.no_grad():

                        # Get Last Value of the Next
                        _, _, _, value = self.agent(self.state)
                        last_value = value.item()

                        # Get Number of Steps Before Cutoff
                        steps_before_cutoff = self.buffer.episode_step

                # Trajectory Ends Normally -> Bootstrap Value of Last State
                else: last_value, steps_before_cutoff = 0, 0

                # Discounted Cumulative Reward and Advantage
                q_vals = self.discount_rewards(self.buffer.episode_rewards + [last_value], self.hparams.gae_gamma)[:-1]
                advantage = self.calc_advantage(self.buffer.episode_rewards, self.buffer.episode_values, last_value)

                # Store Trajectory Tuple in Batch Buffer
                self.buffer.append_trajectory((self.buffer.episode_rewards, advantage, q_vals))

                # Reset Episode Parameters
                self.buffer.reset_episode()

                # Reset Environment
                obs, _ = self.env.reset()
                self.state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            if epoch_end:

                # Yield Trajectory Data (States, Actions, Log Probabilities, Q-Values, Advantage)
                for state, action, old_log_prob, q_value, advantage in self.buffer.train_data:
                    yield state, action, old_log_prob, q_value, advantage

                # Save Average Reward
                self.buffer.avg_reward = sum(self.buffer.epoch_rewards) / self.hparams.steps_per_epoch

                # If Epoch Ended Abruptly, Exclude Last Cut-Short Episode to Prevent Stats Skewness
                epoch_rewards = self.buffer.epoch_rewards[:-1] if not done else self.buffer.epoch_rewards

                # Compute Average Episode Reward (Total Reward / Episode Length) and Average Episode Length
                self.buffer.avg_ep_reward = sum(epoch_rewards) / len(epoch_rewards)
                self.buffer.avg_ep_len = (self.hparams.steps_per_epoch - steps_before_cutoff) / len(epoch_rewards)

                # Clear Buffer Batch Data
                self.buffer.reset_batch()

    def train_dataloader(self) -> DataLoader: #():

        """ Initialize the Replay Buffer Dataset used for Retrieving Experiences """

        # Create a Dataset from the ExperienceSourceDataset
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)

        # Create a DataLoader -> Fetch the Data from Dataset into Training Process
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)

        return dataloader

    def configure_optimizers(self):

        # Instantiate Optimizer
        optim = getattr(torch.optim, self.hparams.optim)

        # Create Optimizer for the Actor and Critic Networks
        actor_optimizer  = optim(self.agent.actor.parameters(),  lr=self.hparams.lr_actor,  eps=self.hparams.epsilon)
        critic_optimizer = optim(self.agent.critic.parameters(), lr=self.hparams.lr_critic, eps=self.hparams.epsilon)
        self.optimizers_list = ['actor_optimizer', 'critic_optimizer']

        return [actor_optimizer, critic_optimizer]

    def manual_optimization_step(self, optimizer:torch.optim.Optimizer, parameters, loss, clip_gradient=True):

        """ Execute Manual Optimizer Step and Gradient Clipping """

        # Execute Optimizer Step and Gradient Clipping
        optimizer.zero_grad()
        self.manual_backward(loss)
        if clip_gradient: torch.nn.utils.clip_grad_norm_(parameters, self.hparams.max_grad_norm)
        optimizer.step()

    def actor_loss(self, state, action, advantage, q_value, old_log_prob) -> torch.Tensor: #():

        """ Compute the Actor Loss """

        # Get the Policy Distribution and Log Probability of the Action
        pi, _ = self.agent.actor(state)
        log_prob = self.agent.actor.get_log_prob(pi, action)

        # Clip the Advantage
        ratio = torch.exp(log_prob - old_log_prob)
        clip_adv = torch.clamp(ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio) * advantage

        # Log Metrics
        with torch.no_grad():

            # Compute KL Divergence and Clip Fraction (How Often the Ratio is Outside the Clipping Range)
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clip_fraction = torch.as_tensor(ratio.gt(1 + self.hparams.clip_ratio) | ratio.lt(1 - self.hparams.clip_ratio), dtype=torch.float32).mean().item()

            # Log the KL Divergence and Clip Fraction
            self.log("metrics/approx_kl", approx_kl, prog_bar=False, on_step=False, on_epoch=True)
            self.log("metrics/kl_divergence", 1.5 * self.hparams.target_kl - approx_kl, prog_bar=False, on_step=False, on_epoch=True)
            self.log("metrics/clip_fraction", clip_fraction, prog_bar=False, on_step=False, on_epoch=True)

        # Compute the Actor Loss
        return -(torch.min(ratio * advantage, clip_adv)).mean()

    def critic_loss(self, state, action, advantage, q_value, old_log_prob) -> torch.Tensor: #():

        """ Compute the Critic Loss """

        # Get the Value Function and the Policy Distribution
        value = self.agent.critic(state).reshape(-1)
        pi, _ = self.agent.actor(state)

        # Compute Returns
        returns = advantage + q_value

        if self.hparams.clip_vloss:

            # Clip the Value Function
            value = q_value + torch.clamp(value - q_value, -self.hparams.clip_ratio, self.hparams.clip_ratio)

        # Compute Value and Entropy Losses
        value_loss = torch.nn.functional.mse_loss(returns, value)
        entropy_loss = - torch.mean(pi.entropy())

        # Compute the Critic Loss
        return self.hparams.entropy_coef * entropy_loss + self.hparams.vloss_coef * value_loss
        # return (q_value - value).pow(2).mean()

    def training_step(self, batch, batch_idx):

        """ Carries Out a Single Update to Actor and Critic Network from a Batch of Replay Buffer """

        # Unpack the Batch of Tuple (state, action, old_log_prob, q_value, advantage)
        state, action, log_prob, q_value, advantage = batch

        # Advantages Normalization
        if self.hparams.adv_normalize: advantage = (advantage - advantage.mean()) / advantage.std()

        # Log the Average Episode Length, Episode Reward and Reward
        self.log("episode/avg_ep_len", self.buffer.avg_ep_len, prog_bar=False, on_step=False, on_epoch=True)
        self.log("episode/avg_ep_reward", self.buffer.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("episode/avg_reward", self.buffer.avg_reward, prog_bar=True, on_step=False, on_epoch=True)

        # Get List of Optimizers
        optimizers = self.optimizers()

        # Optimizer Step for 'optim_update' Number of Iterations
        for _ in range(self.hparams.optim_update):

            # Get Actor and Critic Optimizers
            actor_opt = optimizers[self.optimizers_list.index('actor_optimizer')]
            critic_opt = optimizers[self.optimizers_list.index('critic_optimizer')]

            # Compute Actor and Critic Losses
            actor_loss = self.actor_loss(state, action, advantage, q_value, log_prob)
            critic_loss = self.critic_loss(state, action, advantage, q_value, log_prob)
            self.log('losses/actor_loss', actor_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('losses/critic_loss', critic_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # Optimizer Step
            self.manual_optimization_step(actor_opt, self.agent.actor.parameters(), actor_loss, self.hparams.clip_gradient)
            self.manual_optimization_step(critic_opt, self.agent.critic.parameters(), critic_loss, self.hparams.clip_gradient)

    def on_train_epoch_start(self):

        # Annealing the Rate
        if self.hparams.anneal_lr:

            # Anneal Actor and Critic Learning Rates
            self.anneal_lr(self.optimizers()[self.optimizers_list.index('actor_optimizer')], self.hparams.lr_actor)
            self.anneal_lr(self.optimizers()[self.optimizers_list.index('critic_optimizer')], self.hparams.lr_critic)
