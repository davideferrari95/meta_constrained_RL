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
