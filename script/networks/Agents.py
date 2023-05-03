import gym, gym.spaces as spaces
from typing import List, Optional, Union, Tuple

import torch, torch.nn as nn
from torch import distributions as TD

# Select Training Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network Creation Function
def create_mlp(input_shape: Tuple[int], output_dim: int, hidden_sizes: Optional[List[int]] = [128, 128], 
               hidden_mod: Optional[nn.Module] = nn.ReLU(), output_mod: Optional[nn.Module] = None):

    ''' Neural Network Creation Function '''

    # No Hidden Layers
    if len(hidden_sizes) <= 0:

        # Only one Linear Layer
        net = [nn.Linear(input_shape[0], output_dim)]

    else:

        # First Layer with ReLU Activation
        net = [nn.Linear(input_shape[0], hidden_sizes[0]), hidden_mod]

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

    def forward(self, states):

        """ Pass Observations through the Actor Network """

        # Un-Normalized Action Probabilities
        logits = self.actor_net(states)

        # Pass the Logits to a Categorical Distribution (Softmax Operation)
        pi = TD.Categorical(logits=logits)

        # Sample the Action in the Rollout Phase
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: TD.Categorical, actions: torch.Tensor):

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

    def forward(self, states):

        """ Pass Observations through the Actor Network """

        # Mean and Std Action Probabilities
        mu = self.actor_net(states)
        std = torch.exp(self.log_std)

        # Create a Normal Distribution
        pi = TD.Normal(loc=mu, scale=std)

        # Sample the Action in the Rollout Phase
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: TD.Normal, actions: torch.Tensor):

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

    def __init__(self, actor_net: nn.Module, critic_net: nn.Module):

        super(ActorCriticAgent, self).__init__()

        # Instance the Actor and Critic Networks
        self.actor_net = actor_net
        self.critic_net = critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str = DEVICE) -> Tuple: #():

        """
        Takes in the Current State and Returns:
        Agents Policy, Sampled Action, Log Probability of the Action, Value of the Given State

        Args:
            states: Current State of the Environment
            device: Device Used for the Current Batch

        Returns:
            Torch Distribution and Randomly Sampled Action
        """

        # Move the State to the Device
        state = state.to(device=device)

        # Get Distribution, Action and Log Probability
        pi, actions = self.actor_net(state)
        log_probs = self.get_log_prob(pi, actions)

        # Get the Value of the State
        value = self.critic_net(state)

        return pi, actions, log_probs, value

    def get_log_prob(self, pi: Union[TD.Categorical, TD.Normal], actions: torch.Tensor) -> torch.Tensor: #():

        """
        Takes in the Current State and Returns:
        Log Probability of the Action under the Distribution

        Args:
            pi:      Torch Distribution
            actions: Actions Taken by the Distribution

        Returns:
            Log Probability of the Action under pi
        """

        return self.actor_net.get_log_prob(pi, actions)


class PPO_Agent(ActorCriticAgent):

    def __init__(self, env:gym.Env, hidden_sizes:Optional[List[int]] = [128,128], hidden_mod:Optional[nn.Module] = nn.Tanh):

        # Create Critic Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        critic = create_mlp(env.observation_space.shape, 1, hidden_sizes, hidden_mod(), nn.Identity())

        # Create Continuous Actor Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        if isinstance(env.action_space, spaces.Box):
            actor_mlp = create_mlp(env.observation_space.shape, env.action_space.shape[0], hidden_sizes, hidden_mod(), nn.Identity())
            actor = ActorContinuous(actor_mlp, env.action_space.shape[0])

        # Create Discrete Actor Network -> 3 Linear Layers with Hyperbolic Tangent Activation Function
        elif isinstance(env.action_space, spaces.Discrete):
            actor_mlp = create_mlp(env.observation_space.shape, env.action_space.n, hidden_sizes, hidden_mod(), nn.Identity())
            actor = ActorCategorical(actor_mlp)

        # Raise Error if Action Space is not of Type Box or Discrete
        else: raise NotImplementedError('Env action space should be of type Box (Continuous) or Discrete (Categorical). '
                                       f'Got type: {type(env.action_space)}')

        # Instance the Actor Critic Agent
        super(PPO_Agent, self).__init__(actor, critic)
