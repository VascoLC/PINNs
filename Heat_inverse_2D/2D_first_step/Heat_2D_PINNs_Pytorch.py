import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math

# ======================================
# Helper Function: reference conductivity
# ======================================
def get_ref_k(u, mod=np):
    return 0.02 * mod.exp(-((u - 0.5) ** 2) * 20)

# ======================================
# Neural Networks
# ======================================
class NetU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(2, 32), nn.Tanh()])
        for _ in range(3):  # 4 hidden layers total
            self.hidden.append(nn.Linear(32, 32))
            self.hidden.append(nn.Tanh())
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

class NetK(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(1, 5), nn.Tanh(), nn.Linear(5, 5), nn.Tanh()])
        self.out = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u):
        u_norm = 2.0 * (u - 0.5)
        x = u_norm
        for layer in self.hidden:
            x = layer(x)
        z = self.sigmoid(self.out(x))
        return 0.1 * z
    

# ======================================
# Data Loading
# ======================================
df = pd.read_csv(r"C:/PINNs_Git/PINNs/odil/examples/heat/2D/First_step/out_heat_inverse_2D_32_IMPOSEDPOINTS/imposed_data_with_u_2D.csv")

XYT_obs_np = df[["x", "y","t"]].values.astype(np.float32)
u_obs_np  = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np  = get_ref_k(u_obs_np).astype(np.float32)

XYT_obs = torch.tensor(XYT_obs_np)
u_obs  = torch.tensor(u_obs_np)
k_obs  = torch.tensor(k_obs_np)

# Initial condition
n_ic = 400
x_ic = torch.rand(n_ic, 1)
y_ic = torch.rand(n_ic, 1)
t_ic = torch.zeros_like(x_ic)
xyt_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
u_ic = torch.exp(-50 * (x_ic - 0.5) ** 2) - math.exp(-50 * 0.5**2)

# Boundary condition

n_bc = 400
x_bc0 = torch.zeros((n_bc,1));y_bc0 = torch.rand(n_bc,1);t_bc0 = torch.rand(n_bc,1)
xyt_bc0 = torch.cat([x_bc0,y_bc0,t_ic])