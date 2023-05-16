import torch, gym
import combined_shape, PPO_Agent, DEVICE

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
    cur_cost_ph = torch.zeros((), torch.float32)

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

