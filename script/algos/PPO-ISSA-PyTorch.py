# Import RL Modules
from networks.Agents import PPO_Agent, DEVICE
from networks.Buffers import ExperienceSourceDataset, BatchBuffer, PPOBuffer
from envs.Environment import create_environment, record_violation_episode
from envs.DefaultEnvironment import custom_environment_config
from utils.Utils import CostMonitor, FOLDER, combined_shape

# Import AdamBA Algorithm
from utils.AdamBA import AdamBA, AdamBA_SC, store_heatmap, quadratic_programming

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
        self.configure_environment(environment_config, seed, record_video, record_epochs)

        # Create PPO Agent (Policy and Value Networks)
        agent_kwargs = dict(reward_penalized=reward_penalized, objective_penalized=objective_penalized, learn_penalty=learn_penalty)
        self.agent = PPO_Agent(self.env, hidden_sizes, getattr(torch.nn, hidden_mod), **agent_kwargs).to(DEVICE)

        # Initialize PPO Buffer
        self.buffer: PPOBuffer = self.create_replay_buffer()

        # Main
        self.init_penalty()
        self.init_optimizers()
        self.main()
        exit(0)

    def configure_environment(self, environment_config, seed, record_video, record_epochs):

        """ Configure Environment """

        # Create Environment
        env_name, env_config = custom_environment_config(environment_config)
        self.env = create_environment(env_name, env_config, seed, record_video, record_epochs)

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

    def compute_penalty_loss(self, episode_cost):

        if self.hparams.penalty_loss: return -self.penalty_param * (episode_cost - self.hparams.cost_lim)
        else: return -self.penalty * (episode_cost - self.hparams.cost_lim)

    def compute_dkl(self, dist: TD.Distribution, mu_old, std_old):

        """ Returns KL-Divergence Between Two Distributions """

        # Compute Old Distribution from Mean and Std
        old_dist = TD.Normal(mu_old, std_old)

        # Compute KL-Divergence
        kl = torch.distributions.kl_divergence(dist, old_dist)
        return torch.mean(kl)

    def compute_actor_loss(self, data):

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
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss_pi, dict(d_kl=d_kl, approx_kl=approx_kl, ent=entropy.item(), clip_frac=clip_frac)

    def compute_critic_loss(self, data):

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
        r, d, c, ep_ret, ep_cost, ep_len = 0, False, 0, 0, 0, 0

        # Initialize AdamBA Variables
        ep_positive_safety_index, ep_projection_cost_max_margin = 0, 0
        ep_projection_cost_max_0, ep_projection_cost_max_0_2 = 0, 0
        ep_projection_cost_max_0_4, ep_projection_cost_max_0_8 = 0, 0
        ep_adaptive_safety_index_max_sigma, ep_adaptive_safety_index_max_0 = 0, 0

        # Initialize Loggers
        true_cost_logger, closest_distance_cost_logger = [], []
        projection_cost_max_margin_logger, projection_cost_max_0_logger = [], []
        projection_cost_argmin_0_logger, projection_cost_argmin_margin_logger = [], []
        index_argmin_dis_change_logger, index_max_projection_cost_change_logger = [], []
        adaptive_safety_index_default_logger, adaptive_safety_index_sigma_0_00_logger = [], []
        index_adaptive_max_change_logger, all_out_logger = [], []
        store_logger, cnt_store_json = False, 0

        # HeatMap Loggers
        store_heatmap_video, store_heatmap_trigger = False, False
        cnt_store_heatmap, cnt_store_heatmap_trigger = 0, 0

        # Initialize Vars
        cur_penalty, cum_cost = 0, 0
        cnt_positive_cost, cnt_valid_adamba, cnt_all_out, cnt_one_valid = 0, 0, 0, 0
        cnt_exception, cnt_itself_satisfy, cnt_timestep = 0, 0, 0

        # Main Training Loop
        for epoch in range(self.hparams.max_epochs):

            print(f'Epoch: {epoch+1}')

            # Get Current Penalty
            if self.agent.use_penalty: cur_penalty = self.penalty

            for t in range(self.local_steps_per_epoch):

                cnt_timestep += 1

                # Get outputs from policy
                # pi, actions, log_probs, value, cost_value = self.agent(o)
                pi, a, logp_t, v_t, vc_t = self.agent(torch.as_tensor(o, dtype=torch.float32))

                # AdamBA Safety  Layer

                # Logging for Plotting: Adaptive Safety-Index
                valid_adamba_sc = "Not activated"
                u_new = None

                # Index
                index_max_1 = self.env.projection_cost_index_max(self.hparams.margin)
                index_argmin_dis_1 = self.env.projection_cost_index_argmin_dis(self.hparams.margin)
                index_adaptive_max_1 = self.env.adaptive_safety_index_index(k=self.hparams.k, n=self.hparams.n, sigma=self.hparams.sigma)

                # Projection Cost Max
                projection_cost_max_margin_logger.append(self.env.projection_cost_max(self.hparams.margin))
                projection_cost_max_0_logger.append(self.env.projection_cost_max(0))

                # Projection Cost Argmin
                projection_cost_argmin_margin_logger.append(self.env.projection_cost_argmin_dis(self.hparams.margin))
                projection_cost_argmin_0_logger.append(self.env.projection_cost_argmin_dis(0))

                # Distribution Cost
                true_cost_logger.append(c)
                closest_distance_cost_logger.append(self.env.closest_distance_cost())

                # Synthesis Safety Index
                safe_index_now = self.env.adaptive_safety_index(k=self.hparams.k, sigma=self.hparams.sigma, n=self.hparams.n)
                adaptive_safety_index_default_logger.append(safe_index_now)
                adaptive_safety_index_sigma_0_00_logger.append(self.env.adaptive_safety_index(k=self.hparams.k, n=self.hparams.n, sigma=0.00))

                trigger_by_pre_execute = False

                # Pre-Execute to Determine Whether `trigger_by_pre_execute` is True
                if self.hparams.pre_execute and safe_index_now < 0: # if current safety index > 0, we should run normal AdamBA

                    stored_state = copy.deepcopy(self.env.sim.get_state())

                    # Simulate the Action in AdamBA
                    s_new = self.env.step(a, simulate_in_adamba=True)

                    # Get the Safety Index
                    safe_index_future = self.env.adaptive_safety_index(k=self.hparams.k, sigma=self.hparams.sigma, n=self.hparams.n)

                    # Check Safe Index
                    if safe_index_future >= self.hparams.pre_execute_coef: trigger_by_pre_execute = True
                    else: trigger_by_pre_execute = False

                    # Reset Env (Set q_pos and q_vel)
                    self.env.sim.set_state(stored_state)

                    """
                    Note that the Position-Dependent Stages of the Computation Must Have Been Executed
                    for the Current State in order for These Functions to Return Correct Results.

                    So to be Safe, do `mj_forward` and then `mj_jac` -> The Jacobians will Correspond to the
                    State Before the Integration of Positions and Velocities Took Place.
                    """

                    # Environment Forward
                    self.env.sim.forward()

                # If Adaptive Safety Index > 0 or Triggered by Pre-Execute
                if self.hparams.adamba_layer and (trigger_by_pre_execute or safe_index_now >= 0):

                    # Increase Counters
                    cnt_positive_cost += 1
                    ep_positive_safety_index += 1

                    # AdamBA for Safety Control
                    if self.hparams.adamba_sc == True:

                        # Run AdamBA SC Algorithm -> dt_ration = 1.0 -> Do Not Rely on Small dt
                        adamba_results = AdamBA_SC(o, a.numpy(), env=self.env, threshold=self.hparams.threshold, dt_ratio=1.0, ctrlrange=self.hparams.ctrlrange,
                                                   margin=self.hparams.margin, adaptive_k=self.hparams.k, adaptive_n=self.hparams.n, adaptive_sigma=self.hparams.sigma,
                                                   trigger_by_pre_execute=trigger_by_pre_execute, pre_execute_coef=self.hparams.pre_execute_coef)

                        # Un-Pack AdamBA Results
                        u_new, valid_adamba_sc, env, all_satisfied_u = adamba_results

                        if store_heatmap_trigger:

                            # Store HeatMap Function
                            self.env, cnt_store_heatmap_trigger =  store_heatmap(self.env, cnt_store_heatmap_trigger, trigger_by_pre_execute,
                                                                                 safe_index_now, self.hparams.threshold, self.hparams.n,
                                                                                 self.hparams.k, self.hparams.sigma, self.hparams.pre_execute)

                        # If AdamBA Status = Success
                        if valid_adamba_sc == "adamba_sc success":

                            # Increase AdamBA Counter
                            cnt_valid_adamba += 1

                            # Step in Environment with AdamBA Action (u_new)
                            o2, r, d, truncated, info = self.env.step(np.array([u_new]))

                        else:

                            # Step in Environment with Agent Action
                            o2, r, d, truncated, info = self.env.step(a)

                            # Increase Other AdamBA Counters
                            if valid_adamba_sc == "all out": cnt_all_out += 1
                            elif valid_adamba_sc == "itself satisfy": cnt_itself_satisfy += 1
                            elif valid_adamba_sc == "exception": cnt_exception += 1

                    # Continuous AdamBA (Half Plane with QP-Solving) (note: not working in safety gym due to non-control-affine)
                    else:

                        # Run AdamBA Algorithm
                        [A, b], valid_adamba = AdamBA(o, a.numpy(), env=self.env, threshold=self.hparams.threshold, dt_ratio=0.1,
                                                      ctrlrange=self.hparams.ctrlrange, margin=self.hparams.margin)

                        if valid_adamba == "adamba success":

                            # Increase Counter
                            cnt_valid_adamba += 1

                            # Set the QP-Objective
                            H, f = np.eye(2, 2), [-a[0][0], -a[0][1]]
                            u_new, status = quadratic_programming(H, f, A, [b], initvals=np.array(a[0]), verbose=False)

                            # Step in Environment
                            o2, r, d, truncated, info = self.env.step(np.array([u_new]))

                        else:

                            # Increase Other AdamBA Counters
                            if valid_adamba == "all out": cnt_all_out += 1
                            elif valid_adamba == "itself satisfy": cnt_itself_satisfy += 1
                            elif valid_adamba == "one valid": cnt_one_valid += 1
                            elif valid_adamba == "exception": cnt_exception += 1

                            # Step in Environment
                            o2, r, d, truncated, info = self.env.step(a)

                else:

                    # Step in Environment
                    o2, r, d, truncated, info = self.env.step(a)

                # Logging
                all_out_logger.append(valid_adamba_sc)
                index_max_2 = self.env.projection_cost_index_max(self.hparams.margin)
                index_argmin_dis_2 = self.env.projection_cost_index_argmin_dis(self.hparams.margin)
                index_adaptive_max_2 = self.env.adaptive_safety_index_index(k=self.hparams.k, n=self.hparams.n, sigma=self.hparams.sigma)                
                index_argmin_dis_change_logger.append(index_argmin_dis_1 != index_argmin_dis_2)
                index_max_projection_cost_change_logger.append(index_max_1 != index_max_2)
                index_adaptive_max_change_logger.append(index_adaptive_max_1 != index_adaptive_max_2)

                # Store Logger only if Cost > 0
                if c > 0: store_logger = True

                # Get Cost from Environment
                c = info.get('cost', 0)

                # Track Cumulative Cost Over Training
                cum_cost += c

                # Reward Penalized Buffer Saving
                if self.agent.reward_penalized:

                    # Compute Total Reward
                    r_total = r - cur_penalty * c / (1 + cur_penalty)

                    # Store in PPO Buffer
                    self.buffer.store(pi, o, a, r_total, v_t, 0, 0, logp_t)

                else:

                    # Not Compute Predicted Cost
                    if self.hparams.cpc == False:

                        # Store in PPO Buffer
                        self.buffer.store(pi, o, a, r, v_t, c, vc_t, logp_t)

                    else:

                        # Compute Predicted Cost
                        adaptive_cost = max(0, env.adaptive_safety_index(k=self.hparams.k, n=self.hparams.n, sigma=self.hparams.sigma))
                        r_hat = r - self.hparams.cpc_coef * adaptive_cost

                        if valid_adamba_sc == "adamba_sc success":

                            # Store AdamBA in PPO Buffer
                            self.buffer.store(pi, o, a, r_hat, v_t, c, vc_t, logp_t)
                            self.buffer.store(pi, o, np.array([u_new]), r, v_t, c, vc_t, logp_t)

                        else:

                            # Store Invalid AdamBA in PPO Buffer
                            self.buffer.store(pi, o, a, r_hat, v_t, c, vc_t, logp_t)
                            self.buffer.store(pi, o, a, r, v_t, c, vc_t, logp_t)

                # TODO: Logger
                # logger.store(VVals=v_t, CostVVals=vc_t)

                # Update Observations
                o = o2

                # Increase Episode Return, Cost and Length
                ep_ret, ep_cost, ep_len = ep_ret + r, ep_cost + c, ep_len + 1

                # Increase Projection Cost
                ep_projection_cost_max_margin += max(0, self.env.projection_cost_max(self.hparams.margin))
                ep_projection_cost_max_0      += max(0, self.env.projection_cost_max(0))
                ep_projection_cost_max_0_2    += max(0, self.env.projection_cost_max(0.2))
                ep_projection_cost_max_0_4    += max(0, self.env.projection_cost_max(0.4))
                ep_projection_cost_max_0_8    += max(0, self.env.projection_cost_max(0.8))

                # Increase Adaptive Safety Index
                ep_adaptive_safety_index_max_sigma += max(0, self.env.adaptive_safety_index(k=self.hparams.k, n=self.hparams.n, sigma=self.hparams.sigma))
                ep_adaptive_safety_index_max_0     += max(0, self.env.adaptive_safety_index(k=self.hparams.k, n=self.hparams.n,sigma=0))

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
                        _, _, _, last_value, last_cost_value = self.agent(torch.as_tensor(o, dtype=torch.float32))

                        # Last Cost Value = 0 if Reward Penalized
                        if self.agent.reward_penalized: last_cost_value = 0

                    # Finish Path
                    self.buffer.finish_path(last_value, last_cost_value)

                    # Only Save Episode Return / Length if Trajectory Finished
                    if terminal:

                        # TODO: Logger
                        # logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        self.current_cost = ep_cost
                        ep_activation_ratio = ep_positive_safety_index / ep_len
                        # logger.store(EpActivationRatio=ep_activation_ratio)
                        # logger.store(EpProjectionCostMaxMargin=ep_projection_cost_max_margin)
                        # logger.store(EpProjectionCostMax0=ep_projection_cost_max_0)
                        # logger.store(EpAdaptiveSafetyIndexMaxSigma=ep_adaptive_safety_index_max_sigma)
                        # logger.store(EpAdaptiveSafetyIndexMax0=ep_adaptive_safety_index_max_0)

                        # logger.store(EpProjectionCostMax0_2=ep_projection_cost_max_0_2)
                        # logger.store(EpProjectionCostMax0_4=ep_projection_cost_max_0_4)
                        # logger.store(EpProjectionCost0_8=ep_projection_cost_0_8)

                    # Reset environment
                    o, _ = self.env.reset()
                    r, d, c, ep_ret, ep_len, ep_cost = 0, False, 0, 0, 0, 0

                    # Reset AdamBA Variables
                    ep_positive_safety_index, ep_projection_cost_max_margin = 0, 0
                    ep_projection_cost_max_0, ep_projection_cost_max_0_2 = 0, 0
                    ep_projection_cost_max_0_4, ep_projection_cost_max_0_8 = 0, 0
                    ep_adaptive_safety_index_max_sigma, ep_adaptive_safety_index_max_0 = 0, 0

                    # AdamBA Store Logger
                    if self.hparams.adamba_layer and self.hparams.adamba_sc:

                        # Store Logger if Violation in Episode
                        if store_logger:

                            print('Violations Found in this Episode')

                            # TODO: ???
                            if cnt_store_json >= 0: pass

                            else:

                                import os, json, time

                                # Increase JSON Counter
                                cnt_store_json += 1
                                print("Violation Episode Coming")
                                print("Storing %d/10 file" %(cnt_store_json))

                                # Logger Dictionary
                                data_all_out = {"closest_distance_cost":closest_distance_cost_logger,
                                                "true_cost": true_cost_logger,
                                                "projection_cost_max_margin": projection_cost_max_margin_logger,
                                                "projection_cost_max_0": projection_cost_max_0_logger,
                                                "all_out": all_out_logger,
                                                "index_argmin_dis_change": index_argmin_dis_change_logger,
                                                "index_max_projection_cost_change": index_max_projection_cost_change_logger,
                                                "index_adaptive_max_change": index_adaptive_max_change_logger,
                                                "adaptive_safety_index_default": adaptive_safety_index_default_logger,
                                                "adaptive_safety_index_sigma_0_00": adaptive_safety_index_sigma_0_00_logger}

                                # Dump Data as JSON
                                json_data_all_out = json.dumps(data_all_out, indent=1)

                                # Prepare JSON File Path
                                time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                                if self.env.constrain_hazards:   json_dir_path = os.path.join(FOLDER, 'data/json_data/fixed_adaptive_hazard_%s_size_%s_threshold_%s_sigma_%s_n_%s_k_%s_pre_execute_%s_fixed_simulation/' % (str(env.hazards_num), str(env.hazards_size), str(self.hparams.threshold), str(self.hparams.sigma), str(self.hparams.n), str(self.hparams.k), str(self.hparams.pre_execute)))
                                elif self.env.constrain_pillars: json_dir_path = os.path.join(FOLDER, 'data/json_data/fixed_adaptive_pillar_%s_size_%s_threshold_%s_sigma_%s_n_%s_k_%s_pre_execute_%s_fixed_simulation/' % (str(env.pillars_num), str(env.pillars_size), str(self.hparams.threshold), str(self.hparams.sigma), str(self.hparams.n), str(self.hparams.k), str(self.hparams.pre_execute)))
                                else: raise NotImplementedError

                                # Make JSON Directory
                                if not os.path.exists(json_dir_path): os.makedirs(json_dir_path)

                                # Write File
                                json_file_path = json_dir_path + '%s.json' % (time_str)
                                with open(json_file_path, 'w') as json_file: json_file.write(json_data_all_out)

                                # Reset Store Logger Flag
                                store_logger = False

                        # else: print('No Violations in this Episode')

                    # Reset Loggers
                    closest_distance_cost_logger, true_cost_logger = [], []
                    projection_cost_max_margin_logger, projection_cost_max_0_logger = [], []
                    projection_cost_argmin_0_logger, projection_cost_argmin_margin_logger = [], []
                    all_out_logger, index_argmin_dis_change_logger = [], []
                    index_max_projection_cost_change_logger, index_adaptive_max_change_logger = [], []
                    adaptive_safety_index_default_logger, adaptive_safety_index_sigma_0_00_logger = [], []

            # TODO: Save Model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)

            # Run Policy Update
            self.update()

            # Cumulative Cost Calculations
            cumulative_cost = cum_cost
            cost_rate = cumulative_cost / ((epoch+1) * self.hparams.steps_per_epoch)

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
            logger.log_tabular('CumulativeCost', cumulative_cost)
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
