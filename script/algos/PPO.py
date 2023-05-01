# Import RL Modules
from networks.Agents import PPO_Agent, DEVICE
from networks.Buffers import ReplayBuffer, RLDataset
from envs.Environment import create_environment, create_vectorized_environment, record_violation_episode
from envs.DefaultEnvironment import custom_environment_config
from utils.Utils import CostMonitor, FOLDER, AUTO

# Import Utilities
import os, sys, gym
from termcolor import colored
from typing import Union, Optional
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
class PPO(LightningModule):

    def __init__(

        # gae = General Advantage Estimation

        self,
        seed:int =-1,
        num_envs:int=1,
        num_steps:int=2048,
        max_epochs:int=1000,
        optim:str='Adam',
        lr:float=3e-4,
        anneal_lr:bool=True,
        epsilon:float=1e-5,
        gae:bool=True,
        gae_gamma:float=0.99,
        gae_lambda:float=0.95,
        num_mini_batches:int=32,
        update_epochs:int=10,
        adv_normalize:bool=True,
        clip_ratio:float=0.2,
        clip_vloss:bool=True,
        entropy_coef:float=0.01,
        vloss_coef:float=0.5,
        max_grad_norm:float=0.5,
        target_kl:float=0.015,


        initial_samples:int=10_000,
        buffer_capacity:int=100_000,
        batch_size:int=2048,
        mini_batch_size:int=64,
        hidden_size:int=256,

        # Environment Configuration Parameters:
        record_video:bool=True, record_epochs:int=100, 
        environment_config:Optional[EnvironmentParams]=None

    ):

        super().__init__()

        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Remove Automatic Optimization (Multiple Optimizers)
        # self.automatic_optimization = False

        # Create Vectorized Environment
        assert num_envs > 0, 'Number of Environments must be Greater than 0'
        env_name, env_config = custom_environment_config(environment_config)
        self.envs = create_vectorized_environment(env_name, env_config, num_envs, seed, record_video, record_epochs)

        # Assert that the Action Space is Continuous
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), 'Only Continuous (Box) Action Space is Supported'

        # Save Environment Config Parameters
        self.EC: Optional[EnvironmentParams] = environment_config

        # Create Violation Environment
        if self.EC.violation_environment: self.violation_env = create_environment(env_name, env_config, seed, record_video, 
                                                          environment_type='violation', env_epochs=self.EC.violation_env_epochs)

        # Create Test Environment
        if self.EC.test_environment: self.test_env = create_environment(env_name, env_config, seed, record_video, 
                                                environment_type='test', env_epochs=self.EC.test_env_epochs)

        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        env_spec = self.envs.envs[0].spec
        self.max_episode_steps = env_spec.max_episode_steps
        assert self.max_episode_steps is not None, (f'self.env.spec.max_episode_steps = None')

        # Set Number of Steps -> If num_steps > max_episode_steps, then num_steps = max_episode_steps
        if num_steps > self.max_episode_steps: num_steps = self.max_episode_steps

        # Create PPO Agent (Policy and Value Networks)
        self.agent = PPO_Agent(self.envs).to(DEVICE)

        # Instantiate Optimizer
        self.optim = getattr(torch.optim, optim)

        # Create Optimizer for the Agent
        self.optimizer = self.optim(self.agent.parameters(), lr=lr, eps=epsilon)

        # Compute Batch Size, Mini-Batch Size, and Number of Updates
        batch_size      = num_steps * num_envs
        mini_batch_size = batch_size // num_mini_batches

        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()

        # FIX: Run PPO Algorithm
        self.ppo_run()

    def ppo_run(self):

        # Storage Setup (Replay Buffer)
        obs       = torch.zeros((self.hparams.num_steps, self.hparams.num_envs) + self.envs.single_observation_space.shape, device=DEVICE)
        actions   = torch.zeros((self.hparams.num_steps, self.hparams.num_envs) + self.envs.single_action_space.shape, device=DEVICE)
        log_probs = torch.zeros((self.hparams.num_steps, self.hparams.num_envs), device=DEVICE)
        rewards   = torch.zeros((self.hparams.num_steps, self.hparams.num_envs), device=DEVICE)
        dones     = torch.zeros((self.hparams.num_steps, self.hparams.num_envs), device=DEVICE)
        values    = torch.zeros((self.hparams.num_steps, self.hparams.num_envs), device=DEVICE)

        global_steps = 0

        # Epochs Update Loop
        for update in range(self.hparams.max_epochs):

            next_obs, _ = self.envs.reset()
            next_obs  = torch.Tensor(next_obs).to(DEVICE)
            next_done = torch.zeros(self.hparams.num_envs, device=DEVICE)

            # Annealing the Rate -> On Epoch Start
            if self.hparams.anneal_lr:

                # Fraction Variable that Linearly Decrease to 0
                frac = 1.0 - update / self.hparams.max_epochs
                lr_now = frac * self.hparams.lr

                # Update the Optimizer Learning Rate
                self.optimizer.param_groups[0]['lr'] = lr_now

            # Policy Rollout -> Play Episode -> On Epoch Start / End
            for step in range(0, self.hparams.num_steps):

                # Increment the Global Steps by the Total Environment Steps
                global_steps += 1 * self.hparams.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():

                    # Get the Next Action-Value
                    action, log_prob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                # Play Episode
                next_obs, reward, done, truncated, info = self.envs.step(action.cpu().numpy())

                # Save Actions, Log Probs, Rewards
                actions[step]  = action
                log_probs[step] = log_prob
                rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)

                # Next Obs and Done
                next_obs, next_done = torch.Tensor(next_obs).to(DEVICE), torch.Tensor(done).to(DEVICE)

                if 'final_info' in info.keys():
                    for item in info['final_info']:
                        if item is not None and 'episode' in item.keys():
                            print(f'epoch={update+1}, global_steps={global_steps}, episodic_return={item["episode"]["r"]}')
                            break

            # Implement GAE -> Calc Advantage Function
            with torch.no_grad():

                next_value = self.agent.get_value(next_obs).reshape(1,-1)

                if self.hparams.gae:

                    advantages = torch.zeros_like(rewards, device=DEVICE)
                    last_gae_lam = 0

                    for t in reversed(range(self.hparams.num_steps)):

                        if t == self.hparams.num_steps - 1:

                            next_non_terminal = 1.0 - next_done
                            next_values = next_value

                        else:

                            next_non_terminal = 1.0 - dones[t + 1]
                            next_values = values[t + 1]

                        delta = rewards[t] + self.hparams.gae_gamma * next_values * next_non_terminal - values[t]
                        advantages[t] = last_gae_lam = delta + self.hparams.gae_gamma * self.hparams.gae_lambda * next_non_terminal * last_gae_lam

                    returns = advantages + values

                else:

                    returns = torch.zeros_like(rewards, device=DEVICE)

                    for t in reversed(range(self.hparams.num_steps)):

                        if t == self.hparams.num_steps - 1:

                            next_non_terminal = 1.0 - next_done
                            next_return = next_value

                        else:

                            next_non_terminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]

                        returns[t] = rewards[t] + self.hparams.gae_gamma * next_non_terminal * next_return    

                    advantages = returns - values

            # Flatten the Batch
            b_obs        = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_log_probs  = log_probs.reshape(-1)
            b_actions    = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns    = returns.reshape(-1)
            b_values     = values.reshape(-1)

            # MiniBatch SGD
            # Requires the Indices of the Batch
            b_inds = np.arange(self.hparams.batch_size)
            clip_fracs = []

            # For Each Samples in Mini-Batch
            for _ in range(self.hparams.update_epochs):

                # Shuffle the Batch
                np.random.shuffle(b_inds)

                # Loop through the Entire Batch
                for start in range(0, self.hparams.batch_size, self.hparams.mini_batch_size):

                    # Get the Mini Batch Indices
                    end = start + self.hparams.mini_batch_size
                    mb_inds = b_inds[start:end]

                    _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    log_ratio = new_log_prob - b_log_probs[mb_inds]
                    ratio = torch.exp(log_ratio)

                    # Debug Variables
                    with torch.no_grad():

                        # KL Divergence to Know how Aggressively the Policy Updates
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()

                        # Clip Fraction measures How Often the Ratio is Outside the Clipping Range
                        clip_fracs += [((ratio - 1.0).abs() > self.hparams.clip_ratio).float().mean().item()]

                    # Advantage Normalization Trick
                    if self.hparams.adv_normalize: mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                    else: mb_advantages = b_advantages[mb_inds]

                    # Policy (Actor) Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio)
                    pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss Clipping
                    new_value = new_value.view(-1)
                    if self.hparams.clip_vloss:

                        v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -self.hparams.clip_ratio, self.hparams.clip_ratio)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()

                    else: 

                        v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    # Entropy Loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.hparams.entropy_coef * entropy_loss + self.hparams.vloss_coef * v_loss

                    # Global Gradient Clipping
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.hparams.max_grad_norm)
                    self.optimizer.step()

                # Early Stopping
                if self.hparams.target_kl is not None:
                    if approx_kl > self.hparams.target_kl: 
                        break

            # Explained Variance tells if the Value Function is a Good Indicator of the Rewards
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # self.log("charts/learning_rate",      optimizer.param_groups[0]['lr'], global_steps)
            # self.log("losses/value_loss",         v_loss.item(), global_steps)
            # self.log("losses/policy_loss",        pg_loss.item(), global_steps)
            # self.log("losses/entropy",            entropy_loss.item(), global_steps)
            # self.log("losses/approx_kl",          approx_kl.item(), global_steps)
            # self.log("losses/clip_frac",          np.mean(clip_fracs), global_steps)
            # self.log("losses/explained_variance", explained_var, global_steps)

        # Close Gym Environments
        self.envs.close()

    @torch.no_grad()
    def play_episode(self, policy=None):

        # Compute Environment Seeding
        seed = np.random.randint(0,2**32)

        # Reset Environment
        obs, _ = self.env.reset(seed=seed)
        done, truncated = False, False

        while not done and not truncated:

            # Select an Action using our Policy (Random Action in the Beginning)
            if policy:

                # Get only the Action, not the Log Probability
                action, _, _ = policy(obs)
                action = action.cpu().detach().numpy()

            # Sample from the Action Space
            else: action = self.env.action_space.sample()

            # Execute Safe Action on the Environment
            next_obs, reward, done, truncated, next_info = self.env.step(action)

            # Save Safe Experience in Replay Buffer
            # self.buffer.append((obs, action, reward, float(done), next_obs))

            # Update Observations
            obs, info = next_obs, next_info

    def forward(self, x):

        # Input: State of Environment | Output: Policy Computed by our Network
        return self.policy(x)

    def configure_optimizers(self):

        # Agent Optimizer
        agent_optimizer = self.optim(self.agent.parameters(), lr=self.hparams.lr, eps=self.hparams.epsilon)

        return [agent_optimizer]

    def train_dataloader(self):

        # Create a Dataset from the ReplayBuffer
        dataset = RLDataset(self.buffer, self.hparams.batch_size)

        # Create a DataLoader -> Fetch the Data from Dataset into Training Process with some Optimization
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

        return dataloader

    def training_step(self, batch, batch_idx):

        pass

    def on_train_epoch_end(self):

        # Play Episode
        # self.play_episode(policy=self.target_policy)
        self.play_episode(policy=None)

        # Log Episode Return
        self.log("episode/Return", self.env.return_queue[-1].item(), on_epoch=True)
