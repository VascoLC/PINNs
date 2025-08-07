import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# It seems to work!!!!!!! Try to change the sigmas, just that, use 5k iterations. Try 15

k = 2
pi = np.pi
sin, cos = np.sin, np.cos

# Ground truth
def u_true(x):
    return np.sin(pi * (x[:, 0]*k)**2) * np.sin(pi * x[:, 1])


def q_true(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return (((-4 * k**4 * pi**2 * x1**2) - pi**2) * sin(k**2 * pi * x1**2) +
          2 * k**2 * pi * cos(k**2 * pi * x1**2)) * sin(pi * x2)


# Geometry: Unit square
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# PDE: -Δu = q(x, y)
def pde(x, y):
    q = y[:, 1:2]
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    return -u_xx - u_yy + q

# Observation data (synthetic)

x_obs = np.random.rand(200, 2)
u_obs = u_true(x_obs)
observe_u = dde.icbc.PointSetBC(x_obs, u_obs, component=0)

x_obs_1 = np.random.rand(200, 2)
q_obs = q_true(x_obs)
observe_q = dde.icbc.PointSetBC(x_obs,q_obs,component=1)

# Dirichlet boundary
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)

# Dataset
data = dde.data.PDE(
    geom,
    pde,
    [bc, observe_u,observe_q],
    num_domain=2000,
    num_boundary=400,
    num_test=2000
)

# MsFFN: Fourier feature PINN
net = dde.nn.MsFFN(
    [2, 50, 50, 50, 2],
    "tanh",
    "Glorot uniform",
    sigmas=[1,5,10]
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1.5, 1, 1, 2])
losshistory, train_state = model.train(iterations=100000)

#model.compile("L-BFGS")
#losshistory_lbfgs, train_state_lbfgs = model.train(iterations=10000)

# Test on a grid
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x1, x2)
x_test = np.vstack((X.flatten(), Y.flatten())).T
y_pred = model.predict(x_test)

u_pred = y_pred[:, 0].reshape(100, 100)
q_pred = y_pred[:, 1].reshape(100, 100)
u_exact = u_true(x_test).reshape(100, 100)
q_exact = q_true(x_test).reshape(100, 100)

# Compute errors
print("L2 error u:", dde.metrics.l2_relative_error(u_exact, u_pred))
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
epochs = range(len(losshistory.loss_train))
total_loss = np.sum(np.array(losshistory.loss_train), axis=1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, total_loss, label="Total Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs (Adam)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

