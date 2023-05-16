import numpy as np
import torch

"""
PyTorch utilities for trust region optimization
"""

def flat_concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def flat_grad(f, params):
    return flat_concat(torch.autograd.grad(f, params, retain_graph=True))

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = torch.zeros_like(g)
    return x, flat_grad(torch.sum(g * x), params)


# TODO: Fix the PyTorch Translation of this function
""" 
def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])
"""

def assign_params_from_flat(x, params):
    splits = torch.split(x, [p.numel() for p in params])
    new_params = [p_new.view(p.shape) for p, p_new in zip(params, splits)]
    for p, p_new in zip(params, new_params):
        p.data.copy_(p_new)

"""
Conjugate gradient
"""

def cg(Ax, b, cg_iters=10):
    x = torch.zeros_like(b)
    r = b.clone() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.clone()
    r_dot_old = torch.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (torch.dot(p, z) + 1e-8)
        x += alpha * p
        r -= alpha * z
        r_dot_new = torch.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x
