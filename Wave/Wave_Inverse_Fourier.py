
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import time

C = dde.Variable(1.5)

def pde(x, y):
    du_tt = dde.grad.hessian(y, x, i=1, j=1)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    return du_tt - C**2 * du_xx

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

observe_x = np.vstack((np.linspace(-1, 1, num=500), np.full((500), 1))).T
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

net = dde.nn.STMsFFN(
    [2, 64, 64, 64, 64, 1],
    "tanh",
    "Glorot uniform",
    sigmas_x= [1,2,5,10],
    sigmas_t = [1,2,5,10]
)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C)
variable = dde.callbacks.VariableValue(C, period=1000)

start_time = time.time()

losshistory, train_state = model.train(iterations=100_000, callbacks=[variable])

end_time = time.time()
elapsed = end_time - start_time
print(f"Training completed in {elapsed:.2f} seconds")

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