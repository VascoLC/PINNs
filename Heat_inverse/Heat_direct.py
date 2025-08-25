import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
from deepxde.backend import tf
from keras import layers, Model
import csv, numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd

# --- Geometry: x in [0, 1], t in [0, 1]
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- Initial condition
def initial_condition(x):
    g = np.exp(-50 * (x[:, 0:1] - 0.5) ** 2)
    return g - np.exp(-50 * (0.5) ** 2)

# --- Boundary and IC
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_b: on_b)
ic = dde.icbc.IC(geomtime, initial_condition, lambda x, on_i: on_i)

#----------------------------------
# Forward Problem to get temp values
#----------------------------------

def k_np(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

def kappa(u):
    return  0.02 * tf.exp(-20.0 * tf.square(u - 0.5))

def pde0(x, u):
    u_t  = dde.grad.jacobian(u, x, i=0, j=1)    # ∂u/∂t
    u_x  = dde.grad.jacobian(u, x, i=0, j=0)    # ∂u/∂x
    k_u  = kappa(u)                             # known conductivity
    q_x  = dde.grad.jacobian(k_u*u_x, x, i=0, j=0)  # ∂/∂x(k(u)u_x)
    return u_t - q_x

net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")

data = dde.data.TimePDE(
    geomtime, pde0, [bc, ic],
    num_domain   = 1024,  
    num_boundary = 400,   
    num_initial  = 400)  

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=100_000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x_test = np.linspace(0, 1, 400)  
t_test = np.linspace(0, 1, 400)   
X, T = np.meshgrid(x_test, t_test)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
U_pred = model.predict(XT)
U_pred = U_pred.reshape(X.shape)

plt.figure(figsize=(7, 15))
cs = plt.contourf(X, T, U_pred, levels=100, cmap="viridis")
plt.colorbar(cs, label="u")
plt.title("PINN-direct prediction $u(x,t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.tight_layout()
plt.show()

x_flat = X.flatten()
t_flat = T.flatten()
u_flat = U_pred.flatten()
k_values = k_np(u_flat)

df_pred = pd.DataFrame({
    "x": x_flat,
    "t": t_flat,
    "u": u_flat,
    "k": k_values,
})

# Save to CSV
df_pred.to_csv("Train_data_1D.csv", index=False)

