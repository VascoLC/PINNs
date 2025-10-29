import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf

k = 2
pi = np.pi
sin, cos = np.sin, np.cos

# Defining inital losses
def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

# Ground truth
def u_true(x):
    return np.sin(pi * (x[:, 0]*k)**2) * np.sin(pi * x[:, 1])


def q_true(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return (((-4 * k**4 * pi**2 * x1**2) - pi**2) * sin(k**2 * pi * x1**2) +
          2 * k**2 * pi * cos(k**2 * pi * x1**2)) * sin(pi * x2)

def true_solution(x):
    u = u_true(x).reshape(-1, 1)
    q = q_true(x).reshape(-1, 1)
    return np.hstack((u, q))


# Geometry: Unit square
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# PDE: -Δu = q(x, y)
def pde(x, y):
    q = y[:, 1:2]
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    return -u_xx - u_yy + q

# Observation data (synthetic)
x_obs = np.random.rand(500, 2)
u_obs = u_true(x_obs)
observe_u = dde.icbc.PointSetBC(x_obs, u_obs, component=0)

x_obs_1 = np.random.rand(500, 2)
q_obs = q_true(x_obs_1) # Use x_obs_1 for q observations
observe_q = dde.icbc.PointSetBC(x_obs_1, q_obs, component=1)

# Dirichlet boundary
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)

# Dataset
data = dde.data.PDE(
    geom,
    pde,
    [bc, observe_u, observe_q],
    num_domain=4096,
    num_boundary=400,
    solution=true_solution,
    num_test=4096,
)

# MsFFN: Fourier feature PINN
net = dde.nn.MsFFN(
    [2, 100, 100, 100, 2],
    "tanh",
    "Glorot uniform",
    sigmas=[1, 15, 30, 50] 
)

model = dde.Model(data, net)

# Defining the initial losses and the weights for the Loss functions
initial_losses = get_initial_loss(model)
loss_weights = 5 / initial_losses

model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 2000, 0.9),)
pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
losshistory, train_state = model.train(iterations=100_000,display_every=250) 


dde.saveplot(losshistory, train_state, issave=True, isplot=True)



# --- SAVE LOSS HISTORY TO CSV ---
print("\nSaving loss history to loss_history.csv...")
# Create a DataFrame with the loss for each component
loss_df = pd.DataFrame(losshistory.loss_train, columns=['loss_pde', 'loss_bc', 'loss_u', 'loss_q'])
# Insert the training step/epoch number at the beginning
loss_df.insert(0, 'step', losshistory.steps)
# Save to a CSV file
loss_df.to_csv("loss_history.csv", index=False)
print("Loss history saved successfully.")
# --------------------------------

# Test on a grid
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x1, x2)
x_test = np.vstack((X.flatten(), Y.flatten())).T
y_pred = model.predict(x_test)

# --- SAVE PREDICTED FIELD TO CSV ---
print("\nSaving predicted field to predicted_field.csv...")
# Combine the x, y coordinates with the predicted u and q values
results_data = np.hstack((x_test, y_pred))
# Create a DataFrame with appropriate column headers
results_df = pd.DataFrame(results_data, columns=['x', 'y', 'u_pred', 'q_pred'])
# Save to a CSV file
results_df.to_csv("predicted_field.csv", index=False)
print("Predicted field saved successfully.")
# -----------------------------------

u_pred = y_pred[:, 0].reshape(100, 100)
q_pred = y_pred[:, 1].reshape(100, 100)
u_exact = u_true(x_test).reshape(100, 100)
q_exact = q_true(x_test).reshape(100, 100)

# Compute errors
print("\nL2 error u:", dde.metrics.l2_relative_error(u_exact, u_pred))
print("L2 error q:", dde.metrics.l2_relative_error(q_exact, q_pred))

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# u
axs[0, 0].imshow(u_exact, extent=[0, 1, 0, 1], origin="lower")
axs[0, 0].set_title("u_true")
axs[0, 1].imshow(u_pred, extent=[0, 1, 0, 1], origin="lower")
axs[0, 1].set_title("u_pred")
axs[0, 2].imshow(np.abs(u_pred - u_exact), extent=[0, 1, 0, 1], origin="lower")
axs[0, 2].set_title("u error")

# q
axs[1, 0].imshow(q_exact, extent=[0, 1, 0, 1], origin="lower")
axs[1, 0].set_title("q_true")
axs[1, 1].imshow(q_pred, extent=[0, 1, 0, 1], origin="lower")
axs[1, 1].set_title("q_pred")
axs[1, 2].imshow(np.abs(q_pred - q_exact), extent=[0, 1, 0, 1], origin="lower")
axs[1, 2].set_title("q error")

plt.tight_layout()
plt.show()

# Get losses
# This part is now handled by the pandas DataFrame above, but we can keep the plot
total_loss = loss_df[['loss_pde', 'loss_bc', 'loss_u', 'loss_q']].sum(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(loss_df['step'], total_loss, label="Total Loss")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs Steps (Adam)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()