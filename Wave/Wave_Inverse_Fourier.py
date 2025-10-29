
import os
os.environ["DDE_BACKEND"] = "pytorch"
import pandas as pd
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import time


def pde(x, y):
    du_tt = dde.grad.hessian(y, x, i=1, j=1)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    return du_tt -  du_xx

def U_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:]
    u_exact = np.zeros_like(x_)
    for k in range(1,6):
        u_exact+=1/10*(np.cos((x_-t_+0.5)*np.pi*k)+np.cos((x_+t_+0.5)*np.pi*k))
    return u_exact

def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, U_exact, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, U_exact, lambda _, on_initial: on_initial)
ic_2 = dde.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda x, _: dde.utils.isclose(x[1], 0),
)

observe_x = np.vstack((np.linspace(-1, 1, num=4096), np.full((4096), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, U_exact(observe_x), component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic,ic_2, observe_y],
    num_domain=4096,
    num_boundary=400,
    num_initial=400,
    solution=U_exact,
    num_test=4096,
)


#net = dde.nn.STMsFFN([2, 100, 100, 100, 1],"tanh","Glorot uniform",sigmas_x= [1],sigmas_t= [1,10])


net = dde.nn.FNN([2] + [100] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
start_time = time.time()
losshistory, train_state = model.train(iterations=100)
model.compile("L-BFGS",metrics=["l2 relative error"])
dde.optimizers.config.set_LBFGS_options(
    maxiter=50000,
    maxfun=50000,
    ftol=1e-12,   # loss tolerance
    gtol=1e-12,   # gradient tolerance
)

losshistory, train_state = model.train()
end_time = time.time()
elapsed = end_time - start_time
print(f"Training completed in {elapsed:.2f} seconds")

# --- SAVE LOSS HISTORY TO CSV ---
print("\nSaving loss history to loss_history.csv...")
# Create a DataFrame with the loss for each component
loss_df = pd.DataFrame(losshistory.loss_train, columns=['loss_pde', 'loss_bc','loss_ic','loss_ic2', 'loss_u'])
# Insert the training step/epoch number at the beginning
loss_df.insert(0, 'step', losshistory.steps)
# Save to a CSV file
loss_df.to_csv("loss_history.csv", index=False)
print("Loss history saved successfully.")

# Save and plot the solution results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x_test = np.linspace(-1, 1, 500)  
t_test = np.linspace(0, 1, 500)   

# Create a meshgrid of x and t values
X, T = np.meshgrid(x_test, t_test)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Get the PINN solution for these points
U_pred = model.predict(XT).reshape(X.shape)  # reshape to match meshgrid shape

# Get the exact solution for these points
U_ref = U_exact(XT).reshape(X.shape)

# Plotting the comparison
plt.figure(figsize=(12, 5))

# Plot the reference solution
plt.subplot(1, 2, 1)
plt.contourf(X, T, U_ref, levels=100, cmap="viridis")
plt.colorbar()
plt.title("Reference Solution with Fourier features $U(x,t)$")
plt.xlabel("x")
plt.ylabel("t")

# Plot the predicted solution (from PINN)
plt.subplot(1, 2, 2)
plt.contourf(X, T, U_pred, levels=100, cmap="viridis")
plt.colorbar()
plt.title("PINN Predicted Solution with Fourier features")
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()

# --- NEW: Save Field Coordinates and Values to CSV ---
print("\nSaving predicted field data...")
# Combine the flattened coordinates (x, t) and the flattened predicted values (U_pred)
field_data = np.hstack((XT, U_pred.flatten()[:, None]))
# Define headers for the CSV file
field_headers = ["x", "t", "U_pred"]
# Create a pandas DataFrame
field_df = pd.DataFrame(field_data, columns=field_headers)
# Save the DataFrame to a CSV file
field_df.to_csv("predicted_field.csv", index=False)
print("Predicted field data saved to predicted_field.csv")
# ----------------------------------------------------