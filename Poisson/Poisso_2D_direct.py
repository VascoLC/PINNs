#!/usr/bin/env python3
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. Constants
# -----------------------------
k = 2
pi = np.pi
sin, cos = tf.sin, tf.cos  

# -----------------------------
# 2. Ground truth definitions
# -----------------------------
def q_true(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return (
        ((-4 * k**4 * pi**2 * x1**2) - pi**2) * sin(k**2 * pi * x1**2)
        + 2 * k**2 * pi * cos(k**2 * pi * x1**2)
    ) * sin(pi * x2)

def u_true_numpy(x_np):
    return np.sin(pi * (x_np[:, 0] * k) ** 2) * np.sin(pi * x_np[:, 1])

# -----------------------------
# 3. Geometry and PDE
# -----------------------------
geom = dde.geometry.Rectangle([0, 0], [1, 1])

def pde(x, y):
    u = y[:, 0:1]
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    
    # Use the known analytical source term
    q = q_true(x)

    return -u_xx - u_yy + q

# -----------------------------
# 4. Boundary and data points
# -----------------------------
bc = dde.icbc.DirichletBC(
    geom, 
    lambda x: 0, 
    lambda _, on_boundary: on_boundary, 
    component=0
)

data = dde.data.PDE(
    geom,
    pde,
    [bc],  
    num_domain=4096,
    num_boundary=400,
    solution=u_true_numpy, 
    num_test=4096,
)

# -----------------------------
# 5. Network and model
# -----------------------------
net = dde.nn.FNN([2, 100, 100, 100, 1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
)

# -----------------------------
# 6. Training
# -----------------------------
losshistory, train_state = model.train(
    iterations=10000, 
    display_every=1000
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# -----------------------------
# 7. Save loss history 
# -----------------------------
loss_df = pd.DataFrame(losshistory.loss_train, columns=["loss_pde", "loss_bc"])
loss_df.insert(0, "step", losshistory.steps)
loss_df.to_csv("loss_history.csv", index=False)
print("Loss history saved successfully.")

# -----------------------------
# 8. Prediction and Saving
# -----------------------------
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x1, x2)
x_test = np.vstack((X.flatten(), Y.flatten())).T

y_pred = model.predict(x_test)
u_pred = np.array(y_pred).reshape(-1, 1)

print("\nSaving predicted field to predicted_field.csv...")
results_data = np.hstack((x_test, u_pred))
results_df = pd.DataFrame(results_data, columns=["x", "y", "u_pred"])
results_df.to_csv("predicted_field.csv", index=False)
print("Predicted field saved successfully.")

# -----------------------------
# 9. Error computation
# -----------------------------
u_exact_np = u_true_numpy(x_test)
u_exact = u_exact_np.reshape(100, 100)
u_pred = u_pred.reshape(100, 100)

print("\nL2 error u:", dde.metrics.l2_relative_error(u_exact, u_pred))

# -----------------------------
# 10. Visualization
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

im0 = axs[0].imshow(u_exact, extent=[0, 1, 0, 1], origin="lower", cmap='viridis')
axs[0].set_title("u_true")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(u_pred, extent=[0, 1, 0, 1], origin="lower", cmap='viridis')
axs[1].set_title("u_pred")
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(np.abs(u_pred - u_exact), extent=[0, 1, 0, 1], origin="lower", cmap='jet')
axs[2].set_title("|u_pred - u_true|")
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()

# -----------------------------
# 11. Loss plot (Corrected)
# -----------------------------
# Corrected: Sum only the columns that exist
total_loss = loss_df[["loss_pde", "loss_bc"]].sum(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(loss_df["step"], total_loss, label="Total Loss")
plt.plot(loss_df["step"], loss_df["loss_pde"], label="PDE Loss")
plt.plot(loss_df["step"], loss_df["loss_bc"], label="BC Loss")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs Steps (Adam)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()