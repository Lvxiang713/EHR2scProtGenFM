# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 19:56:50 2025

@author: LvXiang
"""
#gaussian_diffusion.py

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchdiffeq import odeint

class GaussianDiffusion:
    def __init__(self):
        # no fixed schedule
        pass

    def sample_t(self, batch_size, device):
        # sample t uniformly in [0,1]
        return torch.rand(batch_size, device=device)

    def mix(self, x0, x1, t):
        # linear interpolation: x_t = (1-t) * x0 + t * x1
        return (1 - t).view(-1, *([1] * (x0.ndim - 1))) * x0 + t.view(-1, *([1] * (x0.ndim - 1))) * x1

    def velocity(self, x0, x1):
        # constant velocity along straight path
        return x1 - x0

    def train_losses(self, model, x_data, cd):
        # x_data: real data batch (x0)
        device = x_data.device
        batch = x_data.size(0)
        # sample base noise x1 ~ N(0,I)
        x1 = torch.randn_like(x_data)
        # sample t ~ Uniform(0,1)
        t = self.sample_t(batch, device)
        # compute x_t and ground-truth velocity
        x_t = self.mix(x_data, x1, t)
        v_gt = self.velocity(x_data, x1)
        # predict v_theta(x_t, t)
        v_pred = model(x_t, t, cd)
        # MSE loss
        return F.mse_loss(v_pred, v_gt)

    @torch.no_grad()
    def sample_euler(self, model, batch_size, cell_num, dims, cd, steps=100):
        # solve ODE dx/dt = v_theta(x, t) from t=1 to t=0 via Euler
        device = next(model.parameters()).device
        x = torch.randn((batch_size, cell_num, dims), device=device)
        t_vals = torch.linspace(1.0, 0.0, steps, device=device)
        dt = t_vals[1] - t_vals[0]
        for t in t_vals:
            t_batch = t.expand(batch_size)
            v = model(x, t_batch, cd)
            x = x + v * dt
        return x
    
    @torch.no_grad()
    def sample(self, model, batch_size, cell_num, dims, cd,
                           atol=1e-5, rtol=1e-5, method='dopri5'):
        device = next(model.parameters()).device
        # solve ODE dx/dt = v_theta(x, t) from t=1 to t=0 
        x0 = torch.randn((batch_size, cell_num, dims), device=device)
        def ode_func(t, x_flat):
            # x_flat: [batch*cell_num, dims]
            x = x_flat.view(batch_size, cell_num, dims)
            t_batch = torch.full((batch_size,), t, device=device)
            v = model(x, t_batch, cd)
            return v.view(-1, dims)
        x0_flat = x0.view(-1, dims)
        t_span   = torch.linspace(1.0, 0.0, steps=2, device=device)
        sol = odeint(
            ode_func,
            x0_flat,
            t_span,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        x_final = sol[-1].view(batch_size, cell_num, dims)
        return x_final
    