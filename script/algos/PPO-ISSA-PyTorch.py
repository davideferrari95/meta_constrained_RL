# Import RL Modules
from networks.Agents import PPO_Agent, DEVICE
from networks.Buffers import ExperienceSourceDataset, BatchBuffer, PPOBuffer
from envs.Environment import create_environment, record_violation_episode
from envs.DefaultEnvironment import custom_environment_config
from utils.Utils import CostMonitor, FOLDER, combined_shape

# Import Utilities
import os, sys, gym, itertools
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
        max_epochs:         int = 1000,                         # Maximum Number of Epochs
        early_stop_metric:  str = 'episode/avg_ep_reward',      # Metric for Early Stopping

        steps_per_epoch:    int = 2048,                         # How Action-State Pairs to Rollout for Trajectory Collection per Epoch
        batch_size:         int = 512,                          # Batch Size for Training
        num_mini_batches:   int = 32,                           # Number of Mini-Batches for Training
        hidden_sizes:       Optional[List[int]] = [128,128],    # Hidden Layer Sizes for Actor and Critic Networks
        hidden_mod:         Optional[str]       = 'Tanh',       # Hidden Layer Activation Function for Actor and Critic Networks

        # Optimization Parameters:
        optim:              str = 'Adam',                       # Optimizer for Critic and Actor Networks
        actor_update:       int = 80,                           # Number of Gradient Descent to Perform on Each Batch        
        critic_update:      int = 80,                           # Number of Gradient Descent to Perform on Each Batch        
        lr_actor:           float = 3e-4,                       # Learning Rate for Actor Network
        lr_critic:          float = 1e-3,                       # Learning Rate for Critic Network
        lr_penalty:         float = 5e-2,                       # Learning Rate for Penalty

        # GAE (General Advantage Estimation) Parameters:
        gae_gamma:          float = 0.99,                       # Discount Factor for GAE
        gae_lambda:         float = 0.95,                       # Advantage Discount Factor (Lambda) for GAE
        cost_gamma:         float = 0.99,                       # Discount Factor for Cost GAE
        cost_lambda:        float = 0.97,                       # Advantage Discount Factor (Lambda) for Cost GAE
        # adv_normalize:      bool  = True,                       # Normalize Advantage Function

        # PPO (Proximal Policy Optimization) Parameters:
        entropy_reg:        float = 0.0,                        # Entropy Regularization for PPO
        # anneal_lr:          bool  = True,                       # Anneal Learning Rate
        # epsilon:            float = 1e-5,                       # Epsilon for Annealing Learning Rate
        clip_ratio:         float = 0.2,                        # Clipping Parameter for PPO
        # clip_gradient:      bool  = True,                       # Clip Gradient SGD
        # clip_vloss:         bool  = True,                       # Clip Value Loss
        # vloss_coef:         float = 0.5,                        # Value Loss Coefficient
        # entropy_coef:       float = 0.01,                       # Entropy Coefficient
        # max_grad_norm:      float = 0.5,                        # Maximum Gradient Norm
        target_kl:          float = 0.01,                       # Target KL Divergence

        # Cost Constraints / Penalties Parameters:
        cost_lim:           int   = 25,                         # Cost Constraint Limit
        penalty_init:       float = 1.0,                        # Initial Penalty for Cost Constraint

        # AdamBA Safety Constrained Parameters:
        margin:             float = 0.4,                        # Margin for Safety Constraint
        threshold:          float = 0.0,                        # Threshold for Safety Constraint
        ctrlrange:          float = 10.0,                       # Control Range for Safety Constraint
        k:                  float = 3.0,                        # K for Safety Constraint
        n:                  float = 1.0,                        # N for Safety Constraint
        sigma:              float = 0.04,                       # Sigma for Safety Constraint
        cpc:                bool  = False,                      # Use CPC Safety Constraint
        cpc_coef:           float = 0.01,                       # CPC Coefficient
        pre_execute:        bool  = False,                      # Use Pre-Execute Safety Constraint
        pre_execute_coef:   float = 0.0,                        # Pre-Execute Coefficient

        # Environment Configuration Parameters:
        seed:               int  = -1,                          # Random Seed for Environment, Torch and Numpy
        record_video:       bool = True,                        # Record Video of the Environment
        record_epochs:      int  = 100,                         # Record Video Every N Epochs
        environment_config: Optional[EnvironmentParams] = None  # Environment Configuration Parameters

    ):

        super().__init__()

        # Properly Utilize Tensor Cores of the CUDA device ('NVIDIA RTX A4000 Laptop GPU')
        torch.set_float32_matmul_precision('high')

        # Configure Environment
        self.configure_environment(environment_config, seed, record_video, record_epochs)

        # Create PPO Agent (Policy and Value Networks)
        self.agent = PPO_Agent(self.env, hidden_sizes, getattr(torch.nn, hidden_mod)).to(DEVICE)

        # Save Hyperparameters in Internal Properties that we can Reference in our Code
        self.save_hyperparameters()
        
        # Get Number of Workers for Data Loader
        # self.num_workers = os.cpu_count()
        self.num_workers = 1
        
        self.objective_penalized = False
        self.reward_penalized = False
        self.learn_penalty = False
        self.penalty_param_loss = False
        self.trust_region = False
        self.first_order = True
        self.use_penalty = False
        self.constrained = False

        self.initialize()
        self.train()

    def configure_environment(self, environment_config, seed, record_video, record_epochs):

        """ Configure Environment """

        # Create Environment
        env_name, env_config = custom_environment_config(environment_config)
        self.env = create_environment(env_name, env_config, seed, record_video, record_epochs)

        # Get max_episode_steps from Environment -> For Safety-Gym = 1000
        self.max_episode_steps = self.env.spec.max_episode_steps
        assert self.max_episode_steps is not None, (f'self.env.spec.max_episode_steps = None')

    def computational_graph_placeholders(self):

        def placeholder_from_space(space):
            
            if isinstance(space, gym.spaces.Box):
                return torch.zeros(combined_shape(None, space.shape), dtype=torch.float32)
            elif isinstance(space, gym.spaces.Discrete):
                return torch.zeros((None,), dtype=torch.int32)
            raise NotImplementedError(f'Bad Space: {space}')

        def placeholders(*args):
            return [torch.zeros(combined_shape(None,dim), dtype=torch.float32) for dim in args]

        # self.ac_kwargs = {}
        # Share information about action space with policy architecture
        # self.ac_kwargs['action_space'] = self.env.action_space

        # Inputs to computation graph from environment spaces
        x_ph = placeholder_from_space(self.env.observation_space)
        a_ph = placeholder_from_space(self.env.action_space)
        
        # Inputs to computation graph for batch data
        adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

        # Inputs to computation graph for special purposes
        surr_cost_rescale_ph = torch.zeros((), torch.float32)
        self.cur_cost_ph = torch.zeros((), torch.float32)

        # Outputs from actor critic
        ac_outs = PPO_Agent(self.env, self.hparams.hidden_sizes, getattr(torch.nn, self.hparams.hidden_mod)).to(DEVICE)
        # pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs
        pi, action, log_probs, value, cost_value = ac_outs

        # Organize placeholders for zipping with data from buffer on updates
        buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
        # buf_phs += values_as_sorted_list(pi_info_phs)

        # Organize symbols we have to compute at each step of acting in env
        get_action_ops = dict(pi=action, v=value, logp_pi=log_probs)
                            # pi_info=pi_info)

        # If agent is reward penalized, it doesn't use a separate value function
        # for costs and we don't need to include it in get_action_ops; otherwise we do.
        
        # TODO: reward penalized agent (Always False)
        # if not(agent.reward_penalized):

        if not(self.reward_penalized):
            get_action_ops['vc'] = cost_value

        # Make a sample estimate for entropy to use as sanity check
        approx_ent = torch.mean(-log_probs)

    def compute_policy_loss(self, logp, logp_old_ph, adv_ph, cadv_ph, ent, d_kl):
        
        ratio = torch.exp(logp - logp_old_ph)
        
        # Surrogate advantage / clipped surrogate advantage
        clipped_adv = True

        # if agent.clipped_adv:
        if clipped_adv:
            min_adv = torch.where(adv_ph > 0, (1 + self.hparams.clip_ratio) * adv_ph, (1 - self.hparams.clip_ratio) * adv_ph)
            surr_adv = torch.mean(torch.min(ratio * adv_ph, min_adv))
        else:
            surr_adv = torch.mean(ratio * adv_ph)
        
        # Surrogate cost
        surr_cost = torch.mean(ratio * cadv_ph)

        # Create policy objective function, including entropy regularization
        pi_objective = surr_adv + self.hparams.entropy_reg * ent
        
        # Possibly include surr_cost in pi_objective
        # if agent.objective_penalized:
        if self.objective_penalized:
            pi_objective -= self.penalty * surr_cost
            pi_objective /= (1 + self.penalty)
        
        # Loss function for pi is negative of pi_objective
        pi_loss = -pi_objective

        # Optimizer-specific symbols
        # if agent.trust_region:
        if self.trust_region:
            
            from utils.TrustRegion import hessian_vector_product, flat_concat, flat_grad, assign_params_from_flat
            
            # TODO: Get Action from Policy
            pi = None
            
            # Symbols needed for CG solver for any trust region method
            # pi_params = get_vars('pi')
            pi_params = pi
            damping_coeff = 0.1
            
            flat_g = flat_grad(pi_loss, pi_params)
            v_ph, hvp = hessian_vector_product(d_kl, pi_params)
            if damping_coeff > 0:
                hvp += damping_coeff * v_ph

            # Symbols needed for CG solver for CPO only
            flat_b = flat_grad(surr_cost, pi_params)

            # Symbols for getting and setting params
            get_pi_params = flat_concat(pi_params)
            set_pi_params = assign_params_from_flat(v_ph, pi_params)

            training_package = dict(flat_g=flat_g,
                                    flat_b=flat_b,
                                    v_ph=v_ph,
                                    hvp=hvp,
                                    get_pi_params=get_pi_params,
                                    set_pi_params=set_pi_params)

        # elif agent.first_order:
        elif self.first_order:
            
            # Optimizer for first-order policy optimization
            train_pi = torch.optim.Adam(pi, lr=self.hparams.lr_actor)
            
            train_pi.zero_grad()
            pi_loss.backward()
            train_pi.step()

            # Prepare training package for agent
            training_package = dict(train_pi=train_pi)

        else:
            raise NotImplementedError
        
        # Provide training package to agent

        training_package.update(dict(pi_loss=pi_loss,
            surr_cost=surr_cost, d_kl=d_kl,
            target_kl=self.hparams.target_kl, cost_lim=self.hparams.cost_lim))
        
        def prepare_update(self, training_package):
            # training_package is a dict with everything we need (and more)
            # to train.
            self.training_package = training_package        

        prepare_update(training_package)

    def trpo_update(self, loss_pi, act, d_kl, surr_cost):
        
        # Optimizer-specific symbols
        # if agent.trust_region:
        if self.trust_region:
            
            from utils.TrustRegion import hessian_vector_product, flat_concat, flat_grad, assign_params_from_flat
            
            # Symbols needed for CG solver for any trust region method
            # pi_params = get_vars('pi')
            damping_coeff = 0.1
            
            flat_g = flat_grad(loss_pi, act)
            v_ph, hvp = hessian_vector_product(d_kl, act)
            
            if damping_coeff > 0:
                hvp += damping_coeff * v_ph

            # Symbols needed for CG solver for CPO only
            flat_b = flat_grad(surr_cost, act)

            # Symbols for getting and setting params
            get_pi_params = flat_concat(act)
            
            # FIX: `assign_params_from_flat` Function
            set_pi_params = assign_params_from_flat(v_ph, act)

            training_package = dict(flat_g=flat_g,
                                    flat_b=flat_b,
                                    v_ph=v_ph,
                                    hvp=hvp,
                                    get_pi_params=get_pi_params,
                                    set_pi_params=set_pi_params)


# ------------------------------------------------------------------------------
#  PyTorch Correct Code
# ------------------------------------------------------------------------------

    def create_replay_buffer(self) -> PPOBuffer:

        """ Create the Replay Buffer for the PPO Algorithm. """

        # Experience buffer
        local_steps_per_epoch = int(self.hparams.steps_per_epoch / self.num_workers)
        # pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}

        # Buffer Size
        size = local_steps_per_epoch * 2 if self.hparams.cpc else local_steps_per_epoch 

        return PPOBuffer(size, self.env.observation_space.shape, self.env.action_space.shape,
                        self.hparams.gae_gamma, self.hparams.gae_lambda,
                        self.hparams.cost_gamma, self.hparams.cost_lambda)

    def init_penalty(self):

        # if agent.use_penalty:
        if self.use_penalty:
            self.penalty_param = torch.nn.Parameter(torch.tensor(np.log(max(np.exp(self.hparams.penalty_init)-1, 1e-8)), dtype=torch.float32))
            self.penalty = torch.nn.functional.softplus(self.penalty_param)

        if self.learn_penalty: self.penalty_optimizer = torch.optim.Adam([self.penalty_param], lr=self.hparams.lr_penalty)

    def init_optimizers(self):

        # Create an Iterator with the Parameters of the Q-Critic and the Safety-Critic
        critic_params = itertools.chain(self.agent.critic.parameters(), self.agent.cost_critic.parameters())

        # Critic and Actor Optimizers
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=self.hparams.lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.hparams.lr_critic)

    def compute_penalty_loss(self, episode_cost):

        # if agent.penalty_param_loss:
        if self.penalty_param_loss: return -self.penalty_param * (episode_cost - self.hparams.cost_lim)
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
        obs, act, adv = data['observations'], data['actions'], data['advantages'],
        pi_mean_old, pi_std_old, logp_old = data['pi_mean'], data['pi_std'], data['log_probs']
        cost_adv, ret, cost_ret = data['cost_advantages'], data['returns'], data['cost_returns']

        # Get the Policy Distribution and Log Probability of the Action
        # pi, logp = ac.pi(obs, act)
        pi, _ = self.agent.actor(obs)
        log_prob = self.agent.actor.get_log_prob(pi, act)
        # log_prob = self.agent.actor.get_log_prob(obs, act)

        # Compute the Log Probability Ratio
        ratio = torch.exp(log_prob - logp_old)

        # Clip the Advantage
        clipped_adv = True

        if clipped_adv:

            # Clipped Surrogate Advantage
            min_adv = torch.where(adv > 0, (1 + self.hparams.clip_ratio) * adv, (1 - self.hparams.clip_ratio) * adv)
            surr_adv = torch.mean(torch.min(ratio * adv, min_adv))
            # clip_adv = torch.clamp(ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio) * adv
            # surr_adv = torch.mean(torch.min(ratio * adv, clip_adv))

        else:

            # Surrogate Advantage
            surr_adv = torch.mean(ratio * adv)

        # Surrogate Cost
        surr_cost = torch.mean(ratio * cost_adv)

        # Entropy of the Policy
        # entropy = pi.entropy().mean().item()
        entropy = pi.entropy().mean()

        # Create Policy Objective Function, Including Entropy Regularization
        pi_objective = surr_adv + self.hparams.entropy_reg * entropy

        # Possibly include surr_cost in pi_objective
        # if agent.objective_penalized:
        if self.objective_penalized:
            pi_objective -= self.penalty * surr_cost
            pi_objective /= (1 + self.penalty)

        # Loss function for pi is negative of pi_objective
        loss_pi = -pi_objective
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        d_kl = self.compute_dkl(pi, pi_mean_old, pi_std_old)
        approx_kl = (logp_old - log_prob).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.hparams.clip_ratio) | ratio.lt(1 - self.hparams.clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss_pi, dict(d_kl=d_kl, approx_kl=approx_kl, ent=ent, clip_frac=clip_frac)

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
        # if agent.reward_penalized:
        if self.reward_penalized: return v_loss
        else: return v_loss + vc_loss

    def update(self):

        data = self.buffer.get()

        def cares_about_cost(use_penalty, constrained):
            # return self.use_penalty or self.constrained
            return use_penalty or constrained

        # cur_cost = logger.get_stats('EpCost')[0]
        c = self.cur_cost - self.hparams.cost_lim
        if c > 0 and cares_about_cost(self.use_penalty, self.constrained):
            print(colored('Warning! Safety constraint is already violated.', 'red'))

        if self.learn_penalty:

            # Compute Penalty Loss
            penalty_loss = self.compute_penalty_loss(data)

            # Update Penalty
            self.penalty_optimizer.zero_grad()
            penalty_loss.backward()
            self.penalty_optimizer.step()

        # Train Policy with Multiple Steps of SGD
        for i in range(self.hparams.actor_update):

            # Compute Policy Loss
            loss_pi, pi_info = self.compute_actor_loss(data)

            # Get KL-Divergence
            d_kl = pi_info['d_kl']

            # Stop Training if KL-Divergence is Too Large
            if d_kl > 1.5 * self.hparams.target_kl:
                print(colored(f'Early stopping at step {i} due to reaching max kl.', 'red'))
                break

            # Take Optimizer Step
            self.actor_optimizer.zero_grad()
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            self.actor_optimizer.step()

        # Value Function Learning
        for i in range(self.hparams.critic_update):

            # Compute Value Loss
            total_value_loss = self.compute_critic_loss(data)

            # Take Optimizer Step
            self.critic_optimizer.zero_grad()
            total_value_loss.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            self.critic_optimizer.step()
