# Import RL Modules
from networks.Agents import PPO_Agent, DEVICE
from networks.Buffers import ExperienceSourceDataset, BatchBuffer, PPOBuffer
from envs.Environment import create_environment, record_violation_episode
from envs.DefaultEnvironment import custom_environment_config
from utils.Utils import CostMonitor, FOLDER

# Import AdamBA Algorithm
from utils.AdamBA import AdamBA, AdamBA_SC, store_heatmap, quadratic_programming
from utils.AdamBA import safety_layer_AdamBA
from utils.AdamBA import AdamBALogger, AdamBAVariables

# Import Utilities
import os, sys, gym
import copy, itertools
import numpy as np
from tqdm import tqdm
from termcolor import colored
from typing import List, Optional, Tuple

# Import Parameters Class
sys.path.append(FOLDER)
from config.config import EnvironmentParams

# Import PyTorch
import torch
import torch.distributions as TD
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# Create PPO Algorithm
class PPO_ISSA_PyTorch(LightningModule):

    def __init__(

        self,

        # Training Parameters:
        max_epochs:             int = 1000,                         # Maximum Number of Epochs
        early_stop_metric:      str = 'episode/avg_ep_reward',      # Metric for Early Stopping

        steps_per_epoch:        int = 2048,                         # How Action-State Pairs to Rollout for Trajectory Collection per Epoch
        batch_size:             int = 512,                          # Batch Size for Training
        num_mini_batches:       int = 32,                           # Number of Mini-Batches for Training
        hidden_sizes:           Optional[List[int]] = [128,128],    # Hidden Layer Sizes for Actor and Critic Networks
        hidden_mod:             Optional[str]       = 'Tanh',       # Hidden Layer Activation Function for Actor and Critic Networks

        # Optimization Parameters:
        optim:                  str = 'Adam',                       # Optimizer for Critic and Actor Networks
        actor_update:           int = 80,                           # Number of Gradient Descent to Perform on Each Batch        
        critic_update:          int = 80,                           # Number of Gradient Descent to Perform on Each Batch        
        lr_actor:               float = 3e-4,                       # Learning Rate for Actor Network
        lr_critic:              float = 1e-3,                       # Learning Rate for Critic Network
        lr_penalty:             float = 5e-2,                       # Learning Rate for Penalty

        # GAE (General Advantage Estimation) Parameters:
        gae_gamma:              float = 0.99,                       # Discount Factor for GAE
        gae_lambda:             float = 0.95,                       # Advantage Discount Factor (Lambda) for GAE
        cost_gamma:             float = 0.99,                       # Discount Factor for Cost GAE
        cost_lambda:            float = 0.97,                       # Advantage Discount Factor (Lambda) for Cost GAE
        # adv_normalize:          bool  = True,                       # Normalize Advantage Function

        # PPO (Proximal Policy Optimization) Parameters:
        entropy_reg:            float = 0.0,                        # Entropy Regularization for PPO
        # anneal_lr:              bool  = True,                       # Anneal Learning Rate
        # epsilon:                float = 1e-5,                       # Epsilon for Annealing Learning Rate
        clip_ratio:             float = 0.2,                        # Clipping Parameter for PPO
        clipped_adv:            bool  = True,                       # Clipped Surrogate Advantage
        # clip_gradient:          bool  = True,                       # Clip Gradient SGD
        # clip_vloss:             bool  = True,                       # Clip Value Loss
        # vloss_coef:             float = 0.5,                        # Value Loss Coefficient
        # entropy_coef:           float = 0.01,                       # Entropy Coefficient
        # max_grad_norm:          float = 0.5,                        # Maximum Gradient Norm
        target_kl:              float = 0.01,                       # Target KL Divergence

        # Cost Constraints / Penalties Parameters:
        cost_lim:               int   = 25,                         # Cost Constraint Limit
        penalty_init:           float = 1.0,                        # Initial Penalty for Cost Constraint

        # AdamBA Safety Constrained Parameters:
        margin:                 float = 0.4,                        # Margin for Safety Constraint
        threshold:              float = 0.0,                        # Threshold for Safety Constraint
        ctrlrange:              float = 10.0,                       # Control Range for Safety Constraint
        k:                      float = 3.0,                        # K for Safety Constraint
        n:                      float = 1.0,                        # N for Safety Constraint
        sigma:                  float = 0.04,                       # Sigma for Safety Constraint
        cpc:                    bool  = False,                      # Use Compute Predicted Cost Safety Constraint
        cpc_coef:               float = 0.01,                       # CPC Coefficient
        pre_execute:            bool  = False,                      # Use Pre-Execute Safety Constraint
        pre_execute_coef:       float = 0.0,                        # Pre-Execute Coefficient

        # Algorithm / Agent Flags
        reward_penalized:       bool = False,                       # Cost / Penalty Inside the Reward
        objective_penalized:    bool = False,                       # Lagrangian Objective Penalized
        learn_penalty:          bool = False,                       # Learn Penalty if Penalized
        penalty_loss:           bool = False,                       # Compute Penalty Loss
        adamba_layer:           bool = True,                        # Use AdamBA Layer
        adamba_sc:              bool = True,                        # Use Safety-Constrained AdamBA

        # Environment Configuration Parameters:
        seed:               int  = -1,                          # Random Seed for Environment, Torch and Numpy
        record_video:       bool = True,                        # Record Video of the Environment
        record_epochs:      int  = 100,                         # Record Video Every N Epochs
        record_first_epoch: bool = False,                       # Record First Epoch
        environment_config: Optional[EnvironmentParams] = None  # Environment Configuration Parameters

    ):

        super().__init__()

        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()

        # Get Number of Workers for Data Loader
        # self.num_workers = os.cpu_count()
        self.num_workers = 1

        # Compute Local Steps per Epoch
        self.local_steps_per_epoch = int(self.hparams.steps_per_epoch / self.num_workers)

        # Configure Environment
        self.configure_environment(environment_config, record_video, record_epochs, record_first_epoch)

        # Create PPO Agent (Policy and Value Networks)
        agent_kwargs = dict(reward_penalized=reward_penalized, objective_penalized=objective_penalized, learn_penalty=learn_penalty)
        self.agent = PPO_Agent(self.env, hidden_sizes, getattr(torch.nn, hidden_mod), **agent_kwargs).to(DEVICE)

        # Initialize PPO Buffer
        self.buffer: PPOBuffer = self.create_replay_buffer()

        # Initialize Logger
        self.adamba_logger = AdamBALogger()

        # Main
        self.init_penalty()
        self.init_optimizers()
        self.main()
        exit(0)

    def configure_environment(self, environment_config, record_video, record_epochs, record_first_epoch):

        """ Configure Environment """

        # Create Environment
        env_name, env_config = custom_environment_config(environment_config)
        self.env = create_environment(env_name, env_config, record_video, record_epochs, record_first_epoch)

        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        self.max_episode_steps = self.env.spec.max_episode_steps
        assert self.max_episode_steps is not None, (f'self.env.spec.max_episode_steps = None')

    def create_replay_buffer(self) -> PPOBuffer:

        """ Create the Replay Buffer for the PPO Algorithm. """

        # Buffer Size
        size = self.local_steps_per_epoch * 2 if self.hparams.cpc else self.local_steps_per_epoch

        # Create PPO Buffer
        return PPOBuffer(size, self.env.observation_space.shape, self.env.action_space.shape,
                        self.hparams.gae_gamma, self.hparams.gae_lambda,
                        self.hparams.cost_gamma, self.hparams.cost_lambda)

    def init_penalty(self):

        if self.agent.use_penalty:
            self.penalty_param = torch.nn.Parameter(torch.tensor(np.log(max(np.exp(self.hparams.penalty_init)-1, 1e-8)), dtype=torch.float32))
            self.penalty = torch.nn.functional.softplus(self.penalty_param)

        if self.agent.learn_penalty: self.penalty_optimizer = torch.optim.Adam([self.penalty_param], lr=self.hparams.lr_penalty)

    def init_optimizers(self):

        # Create an Iterator with the Parameters of the Q-Critic and the Safety-Critic
        critic_params = itertools.chain(self.agent.critic.parameters(), self.agent.cost_critic.parameters())

        # Critic and Actor Optimizers
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=self.hparams.lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.hparams.lr_critic)

    def compute_penalty_loss(self, episode_cost) -> torch.Tensor:

        if self.hparams.penalty_loss: return -self.penalty_param * (episode_cost - self.hparams.cost_lim)
        else: return -self.penalty * (episode_cost - self.hparams.cost_lim)

    def compute_dkl(self, dist: TD.Distribution, mu_old, std_old):

        """ Returns KL-Divergence Between Two Distributions """

        # Compute Old Distribution from Mean and Std
        old_dist = TD.Normal(mu_old, std_old)

        # Compute KL-Divergence
        kl = torch.distributions.kl_divergence(dist, old_dist)
        return torch.mean(kl)

    def compute_actor_loss(self, data) -> Tuple[torch.Tensor, dict]:

        """ Compute the Actor Loss """

        # Unpack Data from Buffer
        obs, act, adv, cost_adv = data['observations'], data['actions'], data['advantages'], data['cost_advantages']
        pi_mean_old, pi_std_old, logp_old = data['pi_mean'], data['pi_std'], data['log_probs']

        # Get the Policy Distribution and Log Probability of the Action
        pi, _ = self.agent.actor(obs)
        log_prob = self.agent.actor.get_log_prob(pi, act)

        # Compute the Log Probability Ratio
        ratio = torch.exp(log_prob - logp_old)

        if self.hparams.clipped_adv:

            # Clipped Surrogate Advantage
            clip_adv = torch.where(adv > 0, (1 + self.hparams.clip_ratio) * adv, (1 - self.hparams.clip_ratio) * adv)
            # clip_adv = torch.clamp(ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio) * adv
            surr_adv = torch.mean(torch.min(ratio * adv, clip_adv))

        else:

            # Surrogate Advantage
            surr_adv = torch.mean(ratio * adv)

        # Surrogate Cost
        surr_cost = torch.mean(ratio * cost_adv)

        # Entropy of the Policy
        entropy = pi.entropy().mean()

        # Create Policy Objective Function, Including Entropy Regularization
        pi_objective = surr_adv + self.hparams.entropy_reg * entropy

        # Possibly Include `surr_cost` in `pi_objective`
        if self.agent.objective_penalized:
            pi_objective -= self.penalty * surr_cost
            pi_objective /= (1 + self.penalty)

        # Loss Function is Negative of `pi_objective`
        loss_pi = -pi_objective

        # Useful Extra Info
        d_kl = self.compute_dkl(pi, pi_mean_old, pi_std_old)
        approx_kl = (logp_old - log_prob).mean().item()
        clipped = ratio.gt(1 + self.hparams.clip_ratio) | ratio.lt(1 - self.hparams.clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32, device=DEVICE).mean().item()

        return loss_pi, dict(d_kl=d_kl, approx_kl=approx_kl, ent=entropy.item(), clip_frac=clip_frac)

    def compute_critic_loss(self, data) -> torch.Tensor:

        """ Compute the Critic-Value Loss """

        # Unpack Data from Buffer
        obs, ret, cost_ret = data['observations'], data['returns'], data['cost_returns']

        # Get the Value Function Predictions
        value = self.agent.critic(obs)
        cost_value = self.agent.cost_critic(obs)

        # Value / Cost Value Losses
        v_loss = torch.mean((ret - value)**2)
        vc_loss = torch.mean((cost_ret - cost_value)**2)

        # If agent uses penalty directly in reward function, don't train a separate
        # value function for predicting cost returns. (Only use one vf for r - p*c.)
        if self.agent.reward_penalized: return v_loss
        else: return v_loss + vc_loss

    def update(self):

        # Get Buffer Data
        data = self.buffer.get()

        # Get Current Cost
        c = self.current_cost - self.hparams.cost_lim
        if c > 0 and self.agent.cares_about_cost:
            print(colored('Warning! Safety constraint is already violated.', 'red'))

        if self.agent.learn_penalty:

            # Compute Penalty Loss
            penalty_loss = self.compute_penalty_loss(data)

            # Update Penalty
            self.penalty_optimizer.zero_grad()
            penalty_loss.backward()
            self.penalty_optimizer.step()

        # Train Policy with Multiple Steps of SGD
        for _ in range(self.hparams.actor_update):

            # Compute Policy Loss
            loss_pi, pi_info = self.compute_actor_loss(data)

            # Get KL-Divergence
            d_kl = pi_info['d_kl']

            # Stop Policy Update if KL-Divergence is Too Large
            if d_kl > 1.5 * self.hparams.target_kl:
                # print(colored(f'Early stopping at step {i} due to reaching max kl.', 'red'))
                break

            # Take Optimizer Step
            self.actor_optimizer.zero_grad()
            loss_pi.backward()
            self.actor_optimizer.step()

        # Value Function Learning
        for _ in range(self.hparams.critic_update):

            # Compute Value Loss
            total_value_loss = self.compute_critic_loss(data)

            # Take Optimizer Step
            self.critic_optimizer.zero_grad()
            total_value_loss.backward()
            self.critic_optimizer.step()

    def main(self):

        # Initialize Variables Environment
        o, _ = self.env.reset()
        obs = torch.tensor(o, dtype=torch.float32, device=DEVICE)
        r, d, c, ep_ret, ep_cost, ep_len = 0, False, 0, 0, 0, 0

        # Initialize AdamBA Variables
        adamba_vars = AdamBAVariables()

        # Main Training Loop
        for epoch in range(self.hparams.max_epochs):

            print(f'Epoch: {epoch+1}')

            # Get Current Penalty
            if self.agent.use_penalty: adamba_vars.cur_penalty = self.penalty

            for t in range(self.local_steps_per_epoch):

                adamba_vars.cnt_timestep += 1

                # Get outputs from policy
                # pi, actions, log_probs, value, cost_value = self.agent(o)
                pi, a, logp_t, v_t, vc_t = self.agent(obs)

                # AdamBA Safety Layer
                o2, r, d, truncated, info, u_new = safety_layer_AdamBA((o, a, r, d, c, ep_ret, ep_cost, ep_len),
                                                        self.env, adamba_vars, self.adamba_logger, self.hparams)

                # Store Logger only if Cost > 0
                if c > 0: self.adamba_logger.store_json = True

                # Get Cost from Environment
                c = info.get('cost', 0)

                # Track Cumulative Cost Over Training
                adamba_vars.cum_cost += c

                # Reward Penalized Buffer Saving
                if self.agent.reward_penalized:

                    # Compute Total Reward
                    r_total = r - adamba_vars.cur_penalty * c / (1 + adamba_vars.cur_penalty)

                    # Store in PPO Buffer
                    self.buffer.store(pi, obs, a, r_total, v_t, 0, 0, logp_t)

                else:

                    # Not Compute Predicted Cost
                    if self.hparams.cpc == False:

                        # Store in PPO Buffer
                        self.buffer.store(pi, obs, a, r, v_t, c, vc_t, logp_t)

                    else:

                        # Compute Predicted Cost
                        adaptive_cost = max(0, self.env.adaptive_safety_index(k=self.hparams.k, n=self.hparams.n, sigma=self.hparams.sigma)[0])
                        r_hat = r - self.hparams.cpc_coef * adaptive_cost

                        if adamba_vars.valid_adamba_sc == "adamba_sc success":

                            # Store AdamBA in PPO Buffer
                            self.buffer.store(pi, obs, a, r_hat, v_t, c, vc_t, logp_t)
                            self.buffer.store(pi, obs, np.array([u_new]), r, v_t, c, vc_t, logp_t)

                        else:

                            # Store Invalid AdamBA in PPO Buffer
                            self.buffer.store(pi, obs, a, r_hat, v_t, c, vc_t, logp_t)
                            self.buffer.store(pi, obs, a, r, v_t, c, vc_t, logp_t)

                # TODO: Logger
                # logger.store(VVals=v_t, CostVVals=vc_t)

                # Update Observations
                o, obs = o2, torch.tensor(o2, dtype=torch.float32, device=DEVICE)

                # Increase Episode Return, Cost and Length
                ep_ret, ep_cost, ep_len = ep_ret + r, ep_cost + c, ep_len + 1

                # Update AdamBA Variables
                adamba_vars.update_episode_variables(self.env, self.hparams)

                # Compute `timeout`, `terminal` and `epoch_ended` Episode
                timeout = ep_len == self.max_episode_steps
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                # Episode Terminal Procedure
                if terminal or epoch_ended:

                    if epoch_ended and not terminal: print(colored(f'WARNING: Trajectory Cut-Off by Epoch at {ep_len} Steps.', 'yellow'), flush=True)

                    # If Trajectory Didn't Reach Terminal State -> Bootstrap Value Target(s)
                    if d and not timeout: last_value, last_cost_value = 0, 0
                    else:

                        # Get Last Value
                        _, _, _, last_value, last_cost_value = self.agent(obs)

                        # Last Cost Value = 0 if Reward Penalized
                        if self.agent.reward_penalized: last_cost_value = 0

                    # Finish Path
                    self.buffer.finish_path(last_value, last_cost_value)

                    # Only Save Episode Return / Length if Trajectory Finished
                    if terminal:

                        # TODO: Logger
                        # logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        self.current_cost = ep_cost
                        ep_activation_ratio = adamba_vars.ep_positive_safety_index / ep_len
                        # logger.store(EpActivationRatio=ep_activation_ratio)
                        # logger.store(EpProjectionCostMaxMargin=adamba_vars.ep_projection_cost_max_margin)
                        # logger.store(EpProjectionCostMax0=adamba_vars.ep_projection_cost_max_0)
                        # logger.store(EpAdaptiveSafetyIndexMaxSigma=adamba_vars.ep_adaptive_safety_index_max_sigma)
                        # logger.store(EpAdaptiveSafetyIndexMax0=adamba_vars.ep_adaptive_safety_index_max_0)

                        # logger.store(EpProjectionCostMax0_2=adamba_vars.ep_projection_cost_max_0_2)
                        # logger.store(EpProjectionCostMax0_4=adamba_vars.ep_projection_cost_max_0_4)
                        # logger.store(EpProjectionCost0_8=adamba_vars.ep_projection_cost_max_0_8)

                    # Reset environment
                    o, _ = self.env.reset()
                    obs = torch.tensor(o, dtype=torch.float32, device=DEVICE)
                    r, d, c, ep_ret, ep_len, ep_cost = 0, False, 0, 0, 0, 0

                    # Reset AdamBA Variables
                    adamba_vars.reset_episode_variables()

                    # AdamBA Store Logger
                    if self.hparams.adamba_layer and self.hparams.adamba_sc: self.adamba_logger.save_json_logger(self.env)

                    # Reset Loggers
                    self.adamba_logger.reset_loggers()

            # TODO: Save Model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)

            # Run Policy Update
            self.update()

            # Cumulative Cost Calculations
            cost_rate = adamba_vars.cum_cost / ((epoch+1) * self.hparams.steps_per_epoch)

            """
            # Loggers
            logger.log_tabular('Epoch', epoch)

            # Performance stats
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpActivationRatio', with_min_and_max=True)
            logger.log_tabular('EpProjectionCostMaxMargin', with_min_and_max=True)
            logger.log_tabular('EpProjectionCostMax0', with_min_and_max=True)
            # logger.log_tabular('EpProjectionCost0_2', with_min_and_max=True)
            # logger.log_tabular('EpProjectionCost0_4', with_min_and_max=True)
            # logger.log_tabular('EpProjectionCost0_8', with_min_and_max=True)

            logger.log_tabular('EpAdaptiveSafetyIndexMaxSigma', with_min_and_max=True)
            logger.log_tabular('EpAdaptiveSafetyIndexMax0', with_min_and_max=True)

            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('CumulativeCost', adamba_vars.cum_cost)
            logger.log_tabular('CostRate', cost_rate)

            # Value function values
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('CostVVals', with_min_and_max=True)

            # Pi loss and change
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)

            # Surr cost and change
            logger.log_tabular('SurrCost', average_only=True)
            logger.log_tabular('DeltaSurrCost', average_only=True)

            # V loss and change
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)

            # Vc loss and change, if applicable (reward_penalized agents don't use vc)
            if not(self.agent.reward_penalized):
                logger.log_tabular('LossVC', average_only=True)
                logger.log_tabular('DeltaLossVC', average_only=True)

            if self.agent.use_penalty or self.save_penalty:
                logger.log_tabular('Penalty', average_only=True)
                logger.log_tabular('DeltaPenalty', average_only=True)
            else:
                logger.log_tabular('Penalty', 0)
                logger.log_tabular('DeltaPenalty', 0)

            # Anything from the agent?
            agent.log()

            # Policy stats
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)

            # Time and steps elapsed
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.hparams.steps_per_epoch)
            logger.log_tabular('Time', time.time()-start_time)

            # Show results!
            logger.dump_tabular()
            """
