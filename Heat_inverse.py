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
    num_domain   = 800,  
    num_boundary = 400,   
    num_initial  = 400)  

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=5000)

x_test = np.linspace(0, 1, 400)  
t_test = np.linspace(0, 1, 400)   
X, T = np.meshgrid(x_test, t_test)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
U_pred = model.predict(XT)
U_pred = U_pred.reshape(X.shape)

plt.figure(figsize=(7, 15))
cs = plt.contourf(X, T, U_pred, levels=100, cmap="viridis")
plt.colorbar(cs, label="u")
plt.title("PINN prediction $u(x,t)$")
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

'''
#----------------------------------
# Inverse Problem
#----------------------------------

# --- Observation points (for training)

rng = np.random.default_rng(42)
x_all = np.linspace(0, 1, 400)
idx = rng.choice(len(x_all), size=200, replace=False)
x_imp = x_all[idx]
t_imp = np.ones_like(x_imp)              # t=1
XT_imp = np.stack([x_imp, t_imp], axis=1)
u_imp = model.predict(XT_imp)
observe_u = dde.icbc.PointSetBC(XT_imp, u_imp, component=0)

# --- Define k(u) as a neural network
class KappaNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [layers.Dense(10, activation="tanh") for _ in range(2)]
        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return 0.1 * self.out(x)  # keep k(u) small and positive

k_net = KappaNet()

# --- PDE with u(x,t) from one net, k(u) from another
def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)  # du/dx
    u_t = dde.grad.jacobian(u, x, i=0, j=1)  # du/dt

    k_val = k_net(u)  # Coupled dependency

    # Add the term: d/dx(k(u) * u_x)
    term1 = dde.grad.jacobian(k_val * u_x, x, i=0, j=0)  # d/dx(k(u) * du/dx)
    
    # Now, the full PDE with the additional term
    return u_t - term1

# --- Main net for u(x,t)
net_u = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")

# --- Data and model
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_u],
    num_domain=1000,
    num_boundary=400,
    num_initial=400,
)

model = dde.Model(data, net_u)
model.compile("adam", lr=0.001, external_trainable_variables=k_net.trainable_variables)
losshistory, train_state = model.train(epochs=100000)

# --- Plot
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

Nx_plot = 400        
Nt_plot = 400        
x_plot = np.linspace(0.0, 1.0, Nx_plot)
t_plot = np.linspace(0.0, 1.0, Nt_plot)
Xg, Tg = np.meshgrid(x_plot, t_plot)   
XT_plot = np.hstack( (Xg.flatten()[:,None],Tg.flatten()[:,None]) )  
U_inv = model.predict(XT_plot)                      
U_inv = U_inv.reshape(Xg.shape) 

plt.figure(figsize=(7,15))
plt.subplot(1,2,1)
cf = plt.contourf(Xg, Tg, U_inv, 100, cmap='inferno')
plt.colorbar(cf, label='u (inverse)')
plt.title('Inverse-PINN solution $u_h(x,t)$')
plt.xlabel('x'); plt.ylabel('t')
plt.show()
'''



