import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RegularGridInterpolator
import os
from scipy.optimize import fmin_l_bfgs_b

def get_ref_k(u, mod=np):
    # Gaussian.
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))

# Neural Nets
class NetU(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(32, activation="tanh") for _ in range(4)]
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

class NetK(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(5, activation="tanh") for _ in range(2)]
        self.out    = tf.keras.layers.Dense(1, activation="sigmoid")
    def call(self, u):

        u_norm = 2.0*(u- 0.5)
        x = u_norm

        for h in self.hidden:
            x = h(x)
        z = self.out(x)
        return 0.1*z
    
# ------------------------------
# Load measurement data at t
# ------------------------------
df = pd.read_csv("C:\PINNs_Git\PINNs\odil\examples\heat\out_heat_inverse_1D_Imposed_solution/imposed_data_with_u.csv")

XT_obs_np = df[["x","t"]].values.astype(np.float32)   # (N, 2)
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32)  # (N, 1)
k_obs_np   = get_ref_k(u_obs_np)

XT_obs = tf.convert_to_tensor(XT_obs_np)
u_obs   = tf.convert_to_tensor(u_obs_np)
k_obs   = tf.convert_to_tensor(k_obs_np)

# Initial condition
n_ic = 400
x_ic = np.random.rand(n_ic,1)
t_ic = np.zeros((n_ic,1), dtype=np.float32)
xt_ic_np = np.hstack([x_ic,t_ic]).astype(np.float32)

u_ic_np = (np.exp(-50*(x_ic-0.5)**2)-np.exp(-50*0.5**2)).astype(np.float32)

xt_ic = tf.convert_to_tensor(xt_ic_np)
u_ic   = tf.convert_to_tensor(u_ic_np)


# Boundary condition

n_bc = 400
t_bc = np.random.rand(n_bc,1).astype(np.float32)
edges = np.random.choice(["left","right"], size=n_bc)

x_bc = np.random.rand(n_bc,1)
x_bc[edges=="left"]   = 0.0
x_bc[edges=="right"]  = 1.0

xt_bc_np = np.hstack([x_bc,t_bc]).astype(np.float32)
u_bc_np   = np.zeros((n_bc,1), dtype=np.float32)

xt_bc = tf.convert_to_tensor(xt_bc_np)
u_bc   = tf.convert_to_tensor(u_bc_np)

# ------------------------------
# Domain samples for PDE residual
# ------------------------------

n_dom = 4096
x_dom = np.random.rand(n_dom,1)
t_dom = np.random.rand(n_dom,1)
xt_dom_np = np.hstack([x_dom, t_dom]).astype(np.float32)
xt_dom = tf.convert_to_tensor(xt_dom_np)

# ------------------------------
# Instantiate models
# ------------------------------

net_u = NetU()
net_k = NetK()

# ------------------------------
# PDE residual
# ------------------------------

def compute_pde_residual(xt):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xt)
            u = net_u(xt)
            k = net_k(u)
        grads = tape1.gradient(u, xt)             # [du/dx, du/dt]
        du_dx, du_dt = grads[:,0:1], grads[:,1:2]
        flux_x = k * du_dx
    flux_xx = tape2.gradient(flux_x, xt)[:,0:1]
    return du_dt - flux_xx 

# ------------------------------
# Total Loss
# ------------------------------

def compute_loss(xt_domain, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs, k_obs):
    # PDE residual loss
    res = compute_pde_residual(xt_domain)
    loss_pde = tf.reduce_mean(tf.square(res))

    # IC loss
    u_pred_ic = net_u(xt_ic)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic))

    # BC loss
    u_pred_bc = net_u(xt_bc)
    loss_bc = tf.reduce_mean(tf.square(u_pred_bc - u_bc))

    # Supervised u at t=
    u_pred_obs = net_u(xt_obs)
    loss_u  = tf.reduce_mean(tf.square(u_pred_obs - u_obs))

    # Supervised k(u) at t=
    k_pred_obs = net_k(u_pred_obs)
    loss_k  = tf.reduce_mean(tf.square(k_pred_obs - k_obs))

    
    total_loss = loss_pde + loss_ic + loss_bc + loss_u 

    return total_loss, loss_u, loss_k
# ------------------------------
# Plot
# ------------------------------
def plot_and_save(net_u, net_k, epoch, folder_name):
    """
    Generates and saves plots for the temperature field u(x,t) and conductivity k(u).
    """
    # 1. Plot the temperature field u(x,t)
    Nx, Nt = 200, 200 # Lower resolution for faster plotting during training
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t)
    XT_grid = np.hstack([X.flatten()[:, None], T.flatten()[:, None]]).astype(np.float32)
    u_pred = net_u(tf.convert_to_tensor(XT_grid), training=False).numpy().reshape(Nt, Nx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for u(x,t)
    cf = ax1.contourf(X, T, u_pred, levels=100, cmap='viridis')
    fig.colorbar(cf, ax=ax1, label="u(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title(f"Temperature Field u(x,t) at Epoch {epoch}")

    # 2. Plot the learned conductivity k(u) vs the true one
    u_range = np.linspace(0, 1, 400).reshape(-1, 1).astype(np.float32)
    k_true = get_ref_k(u_range)
    k_learned = net_k(tf.convert_to_tensor(u_range), training=False).numpy()
    
    ax2.plot(u_range, k_true, 'r--', linewidth=2, label="True k(u)")
    ax2.plot(u_range, k_learned, 'b-', linewidth=2, label="Learned k(u)")
    ax2.set_xlabel("u")
    ax2.set_ylabel("k(u)")
    ax2.set_title(f"Conductivity k(u) at Epoch {epoch}")
    ax2.legend()
    ax2.grid(True)

    # Save the combined figure
    plt.tight_layout()
    # Use zfill to pad epoch number with zeros for correct file sorting
    filename = os.path.join(folder_name, f"results_epoch_{str(epoch).zfill(6)}.png")
    plt.savefig(filename)
    plt.close(fig) # Close the figure to free up memory
    print(f"--- Saved plots to {filename} ---")

# ------------------------------
# Optimize
# ------------------------------

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
epochs = 200_000

loss_history = [] 

# NEW: Create the results directory if it doesn't exist
results_folder = 'Results Optimisation'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

start_time = time.perf_counter()

# NEW: Plot initial state before training
print("Plotting initial random state (Epoch 0)...")
plot_and_save(net_u, net_k, 0, results_folder)

for epoch in range(1, epochs+1):

    with tf.GradientTape() as tape:
        total_loss, loss_u, loss_k = compute_loss(
            xt_domain = xt_dom,
            xt_bc     = xt_bc, u_bc = u_bc,
            xt_ic     = xt_ic, u_ic = u_ic,
            xt_obs    = XT_obs, u_obs = u_obs, k_obs = k_obs,
        )
    grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables)
    optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables))
    loss_history.append([epoch, total_loss.numpy(), loss_u.numpy(), loss_k.numpy()])

    if epoch % 100 == 0:
        print(f"Epoch {epoch:04d} — Total Loss: {total_loss.numpy():.4e} | Loss_u: {loss_u.numpy():.4e} | Loss_k: {loss_k:.4e}")
    # NEW: Call the plotting function every 500 epochs
    if epoch % 500 == 0:
        plot_and_save(net_u, net_k, epoch, results_folder)
        


end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** Training completed in {elapsed:.2f} seconds. ***")

# ------------------------------
# Plot and Save into csv
# ------------------------------

Nx = 500
Nt = 500

x_values = np.linspace(0, 1, Nx)
t_values = np.linspace(0, 1, Nt)
Xg, Tg   = np.meshgrid(x_values, t_values)

XT_grid = np.hstack([Xg.flatten()[:,None], Tg.flatten()[:,None]]).astype(np.float32)

XT_tensor = tf.convert_to_tensor(XT_grid)
u_pred = net_u(XT_tensor, training=False).numpy().reshape(Nt, Nx)
k_pred = net_k(net_u(XT_tensor, training=False), training=False).numpy().reshape(Nt, Nx)

df_final = pd.DataFrame({
    "x": Xg.flatten(),
    "t": Tg.flatten(),
    "u": u_pred.flatten(),
    "k": k_pred.flatten(),
})

df_final.to_csv("Heat_Inverse_1D_solution_NoBatching_NoK.csv", index=False)

df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv("training_loss_log_1D_solution_NoBatching_NoK.csv", index=False)

plt.figure(figsize=(7, 15))
cf = plt.contourf(Xg, Tg, u_pred, levels=100, cmap="viridis")
plt.colorbar(cf, label="u(x,t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN solution $u(x,t)$")
plt.tight_layout()
plt.show()
