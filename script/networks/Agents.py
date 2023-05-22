import gym, gym.spaces as spaces
from copy import deepcopy
from typing import List, Optional, Union, Tuple

import torch, torch.nn as nn
from torch import distributions as TD

# Import Device
from networks.Utils import DEVICE

# Neural Network Creation Function
def create_mlp(input_dim: int, output_dim: int, hidden_sizes: List[int] = [128, 128], 
               hidden_mod: Optional[nn.Module] = nn.ReLU(), output_mod: Optional[nn.Module] = nn.Identity()) -> nn.Sequential:

    ''' Neural Network Creation Function '''

    # No Hidden Layers
    if len(hidden_sizes) <= 0:

        # Only one Linear Layer
        net = [nn.Linear(input_dim, output_dim)]

    else:

        # First Layer with ReLU Activation
        net = [nn.Linear(input_dim, hidden_sizes[0]), hidden_mod]

        # Add the Hidden Layers
        for i in range(1, len(hidden_sizes)):
            net += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i]), hidden_mod]

        # Add the Output Layer
        net.append(nn.Linear(hidden_sizes[-1], output_dim))

    if output_mod is not None:
        net.append(output_mod)

    # Create a Sequential Neural Network
    return nn.Sequential(*net)

class ActorCategorical(nn.Module):

    """
    Policy Network, for Discrete Action Spaces.
    Returns a Distribution and an Action given an Observation
    """

    def __init__(self, actor_net):

        """
        Args:
            input_shape: Observation Shape of the Environment
            n_actions:   Number of Discrete Actions available in the Environment
        """

        super().__init__()

        # Instance the Actor Network
        self.actor_net = actor_net

    def forward(self, states) -> Tuple[TD.Categorical, torch.Tensor]:

        """ Pass Observations through the Actor Network """

        # Un-Normalized Action Probabilities
        logits = self.actor_net(states)

        # Pass the Logits to a Categorical Distribution (Softmax Operation)
        pi = TD.Categorical(logits=logits)

        # Sample the Action in the Rollout Phase
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: TD.Categorical, actions: torch.Tensor) -> torch.Tensor:

        """
        Takes in a Distribution and Actions and Returns log_prob of Actions under the Distribution

        Args:
            pi:      Torch Distribution
            actions: Actions Taken by Distribution

        Returns:
            Log Probability of the Action under pi
        """

        return pi.log_prob(actions)

class ActorContinuous(nn.Module):

    """
    Policy Network, for Continuous Action Spaces.
    Returns a Distribution and an Action given an Observation
    """

    def __init__(self, actor_net, act_dim):

        """
        Args:
            input_shape: Observation Shape of the Environment
            n_actions:   Number of Discrete Actions available in the Environment
        """

        super().__init__()

        # Instance the Actor Network
        self.actor_net = actor_net

        # Log Standard Deviation Parameter
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, states) -> Tuple[TD.Normal, torch.Tensor]:

        """ Pass Observations through the Actor Network """

        # Mean and Std Action Probabilities
        mu = self.actor_net(states)
        std = torch.exp(self.log_std)

        # Create a Normal Distribution
        pi = TD.Normal(loc=mu, scale=std)

        # Sample the Action in the Rollout Phase
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: TD.Normal, actions: torch.Tensor) -> torch.Tensor:

        """
        Takes in a Distribution and Actions and Returns log_prob of Actions under the Distribution

        Args:
            pi:      Torch Distribution
            actions: Actions Taken by Distribution

        Returns:
            Log Probability of the Action under pi
        """

        return pi.log_prob(actions).sum(axis=-1)

class ActorCriticAgent(nn.Module):

    """
    Actor Critic Agent used during Trajectory Collection.
    It returns a Distribution and an Action given an Observation. 
    """

    # https://github.com/Shmuma/ptan/blob/master/ptan/agent.py

    def __init__(self, actor_net: Union[ActorContinuous, ActorCategorical], critic_net: nn.Module, cost_critic_net: nn.Module):

        super(ActorCriticAgent, self).__init__()

        # Instance Actor and Critic Networks
        self.actor = actor_net
        self.critic = critic_net
        self.cost_critic = cost_critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Takes in the Current State and Returns:
        Agents Policy, Sampled Action, Log Probability of the Action, Value of the Given State

        Args:
            states: Current State of the Environment
            device: Device Used for the Current Batch

        Returns:
            Torch Distribution and Randomly Sampled Action:
            pi, actions, log_probs, value, cost_value
        """

        # Get Distribution, Action and Log Probability
        pi, actions = self.actor(state)
        log_probs = self.get_log_prob(pi, actions)

        # Get Value and Cost-Value of the State
        value = self.critic(state)
        cost_value = self.cost_critic(state)

        return pi, actions, log_probs, value, cost_value

    def get_log_prob(self, pi: Optional[Union[TD.Categorical, TD.Normal]], actions: torch.Tensor, states: Optional[torch.Tensor] = None) -> torch.Tensor:

        """
        Takes in the Current Distribution (or State) and Action and Returns:
        Log Probability of the Action under the Distribution

        Args:
            pi:      Torch Distribution
            states:  States to Pass Through the Network to get the Distribution
            actions: Actions Taken by the Distribution

        Returns:
            Log Probability of the Action under the Distribution
        """

        # Get the Distribution
        if states is not None: pi, _ = self.actor(states)

        # Get the Log Probability of the Action under the Distribution
        return self.actor.get_log_prob(pi, actions)

class PPO_Agent(ActorCriticAgent):

    def __init__(self, env:gym.Env, hidden_sizes:List[int] = [128,128], hidden_mod:nn.Module = nn.Tanh, **kwargs):

        # Create Critic Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        critic = create_mlp(env.observation_space.shape[0], 1, hidden_sizes, hidden_mod(), nn.Identity())
        cost_critic = create_mlp(env.observation_space.shape[0], 1, hidden_sizes, hidden_mod(), nn.Identity())

        # Create Continuous Actor Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        if isinstance(env.action_space, spaces.Box):
            actor_mlp = create_mlp(env.observation_space.shape[0], env.action_space.shape[0], hidden_sizes, hidden_mod(), nn.Identity())
            actor = ActorContinuous(actor_mlp, env.action_space.shape[0])

        # Create Discrete Actor Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        elif isinstance(env.action_space, spaces.Discrete):
            actor_mlp = create_mlp(env.observation_space.shape[0], env.action_space.n, hidden_sizes, hidden_mod(), nn.Identity())
            actor = ActorCategorical(actor_mlp)

        # Raise Error if Action Space is not of Type Box or Discrete
        else: raise NotImplementedError('Env action space should be of type Box (Continuous) or Discrete (Categorical). '
                                       f'Got type: {type(env.action_space)}')

        # Properties
        self.params = deepcopy(kwargs)
        self.params['first_order']  = True
        self.params['trust_region'] = False

        # Instance the Actor Critic Agent
        super(PPO_Agent, self).__init__(actor, critic, cost_critic)

    def ensure_satisfiable_penalty_use(self):
        assert not(self.params.get('reward_penalized') and self.params.get('objective_penalized')), 'Can only use either reward_penalized OR objective_penalized, not both.'
        if not(self.params.get('reward_penalized') or self.params.get('objective_penalized')):
            assert not(self.params.get('learn_penalty')), 'If you are not using a penalty coefficient, you should not try to learn one.'

    def ensure_satisfiable_optimization(self):
        assert not(self.params.get('first_order') and self.params.get('trust_region')), 'Can only use either first_order OR trust_region, not both.'

    @property
    def cares_about_cost(self):
        return self.use_penalty or self.constrained

    @property
    def trust_region(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('trust_region')

    @property
    def first_order(self):
        self.ensure_satisfiable_optimization()
        return self.params.get('first_order')

    @property
    def constrained(self):
        return False

    @property
    def learn_penalty(self):
        return self.params.get('learn_penalty')

    @property
    def use_penalty(self):
        return self.params.get('reward_penalized') or self.params.get('objective_penalized')

    @property
    def objective_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('objective_penalized')

    @property
    def reward_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get('reward_penalized')
