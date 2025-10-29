import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def get_ref_k(u, mod=np):
    # Gaussian.
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))

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
df = pd.read_csv("C:/PINNs_Git/PINNs/odil/examples/heat/out_heat_inverse_2D_32_IMPOSEDPOINTS/imposed_data_with_u_2D.csv")
XYT_obs_np = df[["x", "y", "t"]].values.astype(np.float32)
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np   = get_ref_k(u_obs_np)
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
n_bc = 400 

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

    total_loss = (loss_pde + loss_ic + loss_bc_dirichlet + loss_bc_neumann +
                  loss_u)

    return total_loss, loss_u, loss_k

# ------------------------------
# Plot Function
# ------------------------------
def plot_and_save_fields(net_u, net_k, epoch, folder_name):
    """
    Generates plots for u(x,y,t) and k(u) vs u at multiple time slices.
    """
    os.makedirs(folder_name, exist_ok=True)
    
    nx = ny = 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

    # --- Setup Figure 1 for Temperature u(x,y,t) ---
    fig_u, axes_u = plt.subplots(1, len(time_slices), figsize=(18, 4), sharey=True)
    fig_u.suptitle(f'Temperature Field u(x,y,t)', fontsize=16)
    
    # --- Setup Figure 2 for k(u) vs u ---
    fig_k, axes_k = plt.subplots(1, len(time_slices), figsize=(18, 4), sharey=True)
    fig_k.suptitle(f'Inferred Conductivity k(u) ', fontsize=16)
    mappable_for_colorbar = None

    # --- Loop through time slices to calculate and plot both ---
    for i, t in enumerate(time_slices):
        T = t * np.ones_like(X)
        XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
        XYT_tensor = tf.convert_to_tensor(XYT_grid)

        # Calculate both u and k from the networks
        u_tensor = net_u(XYT_tensor, training=False)
        k_tensor = net_k(u_tensor, training=False)

        u_field = u_tensor.numpy().reshape(nx, ny)
        k_field = k_tensor.numpy().reshape(nx, ny)

    
        # --- Plot temperature field on Figure 1 ---
        ax_u = axes_u[i]
        im_u = ax_u.contourf(X, Y, u_field, levels=100, cmap='YlOrBr', vmin=0, vmax=1)

        # 2. Capture the data from the first plot (t=0) only <--
        if i == 0:
            mappable_for_colorbar = im_u
        ax_u.set_title(f't = {t:.2f}')
        ax_u.set_xlabel('x')
        if i == 0:
            ax_u.set_ylabel('y')
        ax_u.axis("equal")

        # --- Plot k(u) vs u scatter on Figure 2 ---
        ax_k = axes_k[i]
        ax_k.plot(u_field.flatten(), k_field.flatten(), label='Inferred k(u)', color='blue')
        
        # Overlay the true k(u) for reference
        u_true_range = np.linspace(np.min(u_field), np.max(u_field), 100)
        k_true = get_ref_k(u_true_range)
        ax_k.plot(u_true_range, k_true, label='Reference k(u)', color='red') 
        
        ax_k.set_title(f't = {t:.2f}')
        ax_k.set_xlabel('u(x,y,t)')
        ax_k.set_xlim(np.min(u_field), np.max(u_field))
        ax_k.set_ylim(0, 0.03)
        ax_k.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax_k.set_ylabel('k(u)')
        if i == len(time_slices) - 1:
            ax_k.legend()

    # --- Finalize and Save Figure 1 (Temperature) ---
    fig_u.subplots_adjust(right=0.85)
    cbar_ax_u = fig_u.add_axes([0.88, 0.15, 0.01, 0.7])
    if mappable_for_colorbar:
        fig_u.colorbar(mappable_for_colorbar, cax=cbar_ax_u, label='u(x,y,t)')
    u_filename = os.path.join(folder_name, f"temperature_epoch_{str(epoch).zfill(6)}.png")
    fig_u.savefig(u_filename)
    plt.close(fig_u)

    # --- Finalize and Save Figure 2 (k vs u) ---
    k_filename = os.path.join(folder_name, f"conductivity_epoch_{str(epoch).zfill(6)}.png")
    fig_k.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    fig_k.savefig(k_filename)
    plt.close(fig_k)
    
    print(f"--- Saved plots for epoch {epoch} ---")


# ------------------------------
# 8) Training loop (Adam)
# ------------------------------

batch_size_obs = 100
num_obs = XYT_obs_np.shape[0]
num_batches = int(np.ceil(num_obs / batch_size_obs))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
epochs = 25000 
loss_history = [] 

start_time = time.perf_counter()

results_folder = 'training_results_5Batch'
print("Plotting initial random state (Epoch 0)...")
plot_and_save_fields(net_u, net_k, 0, results_folder)

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
        df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
        df_loss.to_csv("training_loss_log_5Batch.csv", index=False)
        print("Saved training_loss_log.csv")
    if epoch % 2500 == 0:
        plot_and_save_fields(net_u, net_k, epoch, results_folder)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** Training completed in {elapsed:.2f} seconds. ***")

# ------------------------------
# 9) Plot and saving
# ------------------------------
print("\nGenerating final high-resolution plots...")
plot_and_save_fields(net_u, net_k, epochs, "final_results_5Batch")

df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv("training_loss_log_5Batch.csv", index=False)
print("Saved training_loss_log.csv")