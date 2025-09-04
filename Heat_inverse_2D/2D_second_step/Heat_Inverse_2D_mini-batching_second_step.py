import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ------------------------------
# 1) Neural Nets
# ------------------------------

class NetU(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(64, activation="tanh") for _ in range(3)]
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

class NetK(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(5, activation="tanh") for _ in range(3)]
        self.out    = tf.keras.layers.Dense(1, activation="sigmoid")
    def call(self, u):

        u_norm = 2.0*(u- 0.5)
        x = u_norm

        for h in self.hidden:
            x = h(x)
        z = self.out(x)
        return 0.1*z

# ------------------------------
# 2) Load measurement data at t
# ------------------------------

# Load all sampled measurements
df = pd.read_csv("data.csv")

XYT_obs_np = df[["x", "y", "t"]].values.astype(np.float32)   
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32) 
k_obs_np   = df[["k"]].values.reshape(-1, 1).astype(np.float32)  

# Convert to tensors
XYT_obs = tf.convert_to_tensor(XYT_obs_np)
u_obs   = tf.convert_to_tensor(u_obs_np)
k_obs   = tf.convert_to_tensor(k_obs_np)

# ------------------------------
# 3) Generate IC and BC samples
# ------------------------------

# 3.1) Initial condition
n_ic = 400
x_ic = np.random.rand(n_ic,1)
y_ic = np.random.rand(n_ic,1)
t_ic = np.zeros((n_ic,1), dtype=np.float32)
xyt_ic_np = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)
g0 = np.exp(-50*(0.5)**2)
u_ic_np   = (np.exp(-50*(x_ic - 0.5)**2) - g0).astype(np.float32)

xyt_ic = tf.convert_to_tensor(xyt_ic_np)
u_ic   = tf.convert_to_tensor(u_ic_np)

# 3.2) Boundary conditions
n_bc = 200 

# Dirichlet BC on x=0 and x=1
# x=0
y_bc0 = np.random.rand(n_bc, 1)
t_bc0 = np.random.rand(n_bc, 1)
x_bc0 = np.zeros((n_bc, 1))
xyt_bc0 = np.hstack([x_bc0, y_bc0, t_bc0])
# x=1
y_bc1 = np.random.rand(n_bc, 1)
t_bc1 = np.random.rand(n_bc, 1)
x_bc1 = np.ones((n_bc, 1))
xyt_bc1 = np.hstack([x_bc1, y_bc1, t_bc1])

tri = np.where(y_bc1 <= 0.5, (y_bc1) / (0.5), (1 - y_bc1) / (0.5))

# Stack for Dirichlet
xyt_bc_dirichlet_np = np.vstack([xyt_bc0, xyt_bc1])
u_bc_dirichlet_0_np = np.zeros((n_bc, 1), dtype=np.float32)
u_bc_dirichlet_1_np = tri.astype(np.float32)
u_bc_dirichlet_np = np.vstack([u_bc_dirichlet_0_np, u_bc_dirichlet_1_np])
xyt_bc_dirichlet = tf.convert_to_tensor(xyt_bc_dirichlet_np.astype(np.float32))
u_bc_dirichlet   = tf.convert_to_tensor(u_bc_dirichlet_np)

# Neumann BC on y=0 and y=1
# y=0
x_bc2 = np.random.rand(n_bc, 1)
t_bc2 = np.random.rand(n_bc, 1)
y_bc2 = np.zeros((n_bc, 1))
xyt_bc2 = np.hstack([x_bc2, y_bc2, t_bc2])
# y=1
x_bc3 = np.random.rand(n_bc, 1)
t_bc3 = np.random.rand(n_bc, 1)
y_bc3 = np.ones((n_bc, 1))
xyt_bc3 = np.hstack([x_bc3, y_bc3, t_bc3])
# Stack for Neumann
xyt_bc_neumann_np = np.vstack([xyt_bc2, xyt_bc3])
xyt_bc_neumann = tf.convert_to_tensor(xyt_bc_neumann_np.astype(np.float32))

# ------------------------------
# 4) Domain samples for PDE residual
# ------------------------------

n_dom = 4096
x_dom = np.random.rand(n_dom,1)
y_dom = np.random.rand(n_dom,1)
t_dom = np.random.rand(n_dom,1)
xyt_dom_np = np.hstack([x_dom, y_dom, t_dom]).astype(np.float32)
xyt_dom = tf.convert_to_tensor(xyt_dom_np)

# ------------------------------
# 5) Instantiate models
# ------------------------------

net_u = NetU()
net_k = NetK()

# ------------------------------
# 6) PDE residual function
# ------------------------------

def compute_pde_residual(xyt):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xyt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xyt)
            u = net_u(xyt)
            k = net_k(u)
        grads = tape1.gradient(u, xyt)             
        du_dx, du_dy, du_dt = grads[:,0:1], grads[:,1:2], grads[:,2:3]
        flux_x = k * du_dx
        flux_y = k * du_dy
    flux_xx = tape2.gradient(flux_x, xyt)[:,0:1]
    flux_yy = tape2.gradient(flux_y, xyt)[:,1:2]
    return du_dt - (flux_xx + flux_yy)

# ------------------------------
# 7) Total loss
# ------------------------------

def compute_loss(xyt_domain, xyt_bc_dirichlet, u_bc_dirichlet, xyt_bc_neumann, xyt_ic, u_ic, xyt_obs, u_obs, k_obs):

    # PDE residual loss
    res = compute_pde_residual(xyt_domain)
    loss_pde = tf.reduce_mean(tf.square(res))

    # IC loss
    u_pred_ic = net_u(xyt_ic)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic))

    # Dirichlet BC and Triangular BC (u=0 at x=0 and tri at x=1)
    u_pred_bc_dirichlet = net_u(xyt_bc_dirichlet)
    loss_bc_dirichlet = tf.reduce_mean(tf.square(u_pred_bc_dirichlet - u_bc_dirichlet))

    # Neumann BC loss (du/dy=0 at y=0 and y=1)
    with tf.GradientTape() as tape_neumann:
        tape_neumann.watch(xyt_bc_neumann)
        u_pred_bc_neumann = net_u(xyt_bc_neumann)
    du_dy_bc = tape_neumann.gradient(u_pred_bc_neumann, xyt_bc_neumann)[:, 1:2]
    loss_bc_neumann = tf.reduce_mean(tf.square(du_dy_bc))

    # Supervised u
    u_pred_obs = net_u(xyt_obs)
    loss_u  = tf.reduce_mean(tf.square(u_pred_obs - u_obs))

    # Supervised k(u)
    k_pred_obs = net_k(u_pred_obs)
    loss_k  = tf.reduce_mean(tf.square(k_pred_obs - k_obs))

    # The loss_k weight is the most important, without it convergence is not achieved so easily
    total_loss = (loss_pde + loss_ic + loss_bc_dirichlet + loss_bc_neumann +
                  loss_u + 100 * loss_k)

    return total_loss, loss_u, loss_k

# ------------------------------
# 8) Training loop (Adam)
# ------------------------------

batch_size_obs = 100
num_obs = XYT_obs_np.shape[0]
num_batches = int(np.ceil(num_obs / batch_size_obs))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
epochs = 5000 # In this case corresponds to 30000 iterations of the optimiser

loss_history = [] 

start_time = time.perf_counter()

for epoch in range(1, epochs+1):
    indices = np.random.permutation(num_obs)

    for batch in range(num_batches):
        batch_idx = indices[batch*batch_size_obs : (batch+1)*batch_size_obs]
        
        XYT_obs_mb = tf.convert_to_tensor(XYT_obs_np[batch_idx])
        u_obs_mb   = tf.convert_to_tensor(u_obs_np[batch_idx])
        k_obs_mb   = tf.convert_to_tensor(k_obs_np[batch_idx])
                
        with tf.GradientTape() as tape:
            total_loss, loss_u, loss_k = compute_loss(
                xyt_domain = xyt_dom,   
                xyt_bc_dirichlet = xyt_bc_dirichlet, u_bc_dirichlet = u_bc_dirichlet,
                xyt_bc_neumann = xyt_bc_neumann,
                xyt_ic = xyt_ic, u_ic = u_ic,
                xyt_obs = XYT_obs_mb, u_obs = u_obs_mb, k_obs = k_obs_mb,
            )
        grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables)
        optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables))
    loss_history.append([epoch, total_loss.numpy(), loss_u.numpy(), loss_k.numpy()])

    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} — Total Loss: {total_loss.numpy():.4e} | Loss_u: {loss_u.numpy():.4e} | Loss_k: {loss_k.numpy():.4e}")

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** Training completed in {elapsed:.2f} seconds. ***")

# ------------------------------
# 9) Plot and saving
# ------------------------------

nx = ny = 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

times_to_save = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

os.makedirs("fields_over_time", exist_ok=True)

# -- Prepare figure --
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i,t in enumerate(times_to_save):
    T = t * np.ones_like(X)

    XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
    XYT_tensor = tf.convert_to_tensor(XYT_grid)

    u_tensor = net_u(XYT_tensor, training=False)
    k_tensor = net_k(u_tensor, training=False)

    u_field = u_tensor.numpy().reshape(nx, ny)
    k_field = k_tensor.numpy().reshape(nx, ny)

    df_final = pd.DataFrame({
        "x": X.flatten(),
        "y": Y.flatten(),
        "t": T.flatten(),
        "u": u_field.flatten(),
        "k": k_field.flatten(),
    })

    df_final.to_csv(f"fields_over_time/field_at_t_{t:.1f}.csv", index=False)
    print(f"Saved field_at_t_{t:.1f}.csv")
        # Plot
    ax = axes[i]
    cf = ax.contourf(X, Y, u_field, levels=100)
    fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04)
    ax.set_title(f"u(x, y, t={t:.1f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()


df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv("training_loss_log.csv", index=False)

'''
# Plot u(x,y,t)
plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, u_field, levels=100)
plt.colorbar(cf, label="u(x,y,t=1)")
plt.title("Predicted Temperature at t = 1")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
'''


# Plot k vs u 
plt.figure(figsize=(5,4))
plt.scatter(u_field, k_field, s=10, alpha=0.6)
plt.xlabel("u"); plt.ylabel("k(u)")
plt.title("Learned Conductivity vs Temperature (t = 1)")
plt.grid(True)
plt.tight_layout()

plt.show()
