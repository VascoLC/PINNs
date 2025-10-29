import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import minimize  # <-- Import SciPy's optimizer

# ------------------------------
# Reference Conductivity
# ------------------------------

def get_ref_k(u, mod=np):
    # Gaussian.
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))

# ------------------------------
# Neural Networks
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
        u_norm = 2.0*(u - 0.5)
        x = u_norm
        for h in self.hidden:
            x = h(x)
        z = self.out(x)
        return 0.1 * z

# ------------------------------
# Load measurement data
# ------------------------------

df = pd.read_csv("C:/PINNs_Git/PINNs/odil/examples/heat/2D/First_step/out_heat_inverse_2D_32_IMPOSEDPOINTS/imposed_data_with_u_2D.csv")
XYT_obs_np = df[["x", "y", "t"]].values.astype(np.float32)
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np   = get_ref_k(u_obs_np)

XYT_obs = tf.convert_to_tensor(XYT_obs_np)
u_obs   = tf.convert_to_tensor(u_obs_np)
k_obs   = tf.convert_to_tensor(k_obs_np)

# ------------------------------
# Generate IC and BC samples
# ------------------------------
n_ic = 400
x_ic = np.random.rand(n_ic,1)
y_ic = np.random.rand(n_ic,1)
t_ic = np.zeros((n_ic,1), dtype=np.float32)
xyt_ic_np = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)
g0 = np.exp(-50*(0.5)**2)
u_ic_np   = (np.exp(-50*(x_ic - 0.5)**2) - g0).astype(np.float32)
xyt_ic = tf.convert_to_tensor(xyt_ic_np)
u_ic   = tf.convert_to_tensor(u_ic_np)

n_bc = 400
y_bc0 = np.random.rand(n_bc, 1); t_bc0 = np.random.rand(n_bc, 1); x_bc0 = np.zeros((n_bc, 1))
xyt_bc0 = np.hstack([x_bc0, y_bc0, t_bc0])
y_bc1 = np.random.rand(n_bc, 1); t_bc1 = np.random.rand(n_bc, 1); x_bc1 = np.ones((n_bc, 1))
xyt_bc1 = np.hstack([x_bc1, y_bc1, t_bc1])
xyt_bc_dirichlet_np = np.vstack([xyt_bc0, xyt_bc1])
u_bc_dirichlet_np = np.zeros((2*n_bc, 1), dtype=np.float32)
xyt_bc_dirichlet = tf.convert_to_tensor(xyt_bc_dirichlet_np.astype(np.float32))
u_bc_dirichlet   = tf.convert_to_tensor(u_bc_dirichlet_np)

x_bc2 = np.random.rand(n_bc, 1); t_bc2 = np.random.rand(n_bc, 1); y_bc2 = np.zeros((n_bc, 1))
xyt_bc2 = np.hstack([x_bc2, y_bc2, t_bc2])
x_bc3 = np.random.rand(n_bc, 1); t_bc3 = np.random.rand(n_bc, 1); y_bc3 = np.ones((n_bc, 1))
xyt_bc3 = np.hstack([x_bc3, y_bc3, t_bc3])
xyt_bc_neumann_np = np.vstack([xyt_bc2, xyt_bc3])
xyt_bc_neumann = tf.convert_to_tensor(xyt_bc_neumann_np.astype(np.float32))

# ------------------------------
# Domain samples for PDE residual
# ------------------------------
n_dom = 4096
x_dom = np.random.rand(n_dom,1)
y_dom = np.random.rand(n_dom,1)
t_dom = np.random.rand(n_dom,1)
xyt_dom_np = np.hstack([x_dom, y_dom, t_dom]).astype(np.float32)
xyt_dom = tf.convert_to_tensor(xyt_dom_np)

# ------------------------------
# Instantiate models
# ------------------------------
net_u = NetU()
net_k = NetK()

# ------------------------------
# PDE residual and loss function
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

def compute_loss(xyt_domain, xyt_bc_dirichlet, u_bc_dirichlet, xyt_bc_neumann, xyt_ic, u_ic, xyt_obs, u_obs, k_obs):
    loss_pde = tf.reduce_mean(tf.square(compute_pde_residual(xyt_domain)))
    loss_ic = tf.reduce_mean(tf.square(net_u(xyt_ic) - u_ic))
    loss_bc_dirichlet = tf.reduce_mean(tf.square(net_u(xyt_bc_dirichlet) - u_bc_dirichlet))


    with tf.GradientTape() as tape_neumann:
        tape_neumann.watch(xyt_bc_neumann)
        u_pred_bc_neumann = net_u(xyt_bc_neumann)
    du_dy_bc = tape_neumann.gradient(u_pred_bc_neumann, xyt_bc_neumann)[:, 1:2]
    loss_bc_neumann = tf.reduce_mean(tf.square(du_dy_bc))


    u_pred_obs = net_u(xyt_obs)
    loss_u = tf.reduce_mean(tf.square(u_pred_obs - u_obs))
    k_pred_obs = net_k(u_pred_obs)
    loss_k = tf.reduce_mean(tf.square(k_pred_obs - k_obs))


    total_loss = (loss_pde + loss_ic + loss_bc_dirichlet + loss_bc_neumann + loss_u + loss_k)
    return total_loss, loss_u, loss_k

# ------------------------------
# Plotting function 
# ------------------------------

def plot_and_save_fields(net_u, net_k, epoch, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    nx = ny = 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig_u, axes_u = plt.subplots(1, len(time_slices), figsize=(18, 4), sharey=True)
    fig_u.suptitle('Temperature Field u(x,y,t)', fontsize=16)

    fig_k, axes_k = plt.subplots(1, len(time_slices), figsize=(18, 4), sharey=True)
    fig_k.suptitle('Inferred Conductivity k(u)', fontsize=16)
    mappable_for_colorbar = None

    for i, t in enumerate(time_slices):
        T = t * np.ones_like(X)
        XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
        XYT_tensor = tf.convert_to_tensor(XYT_grid)
        u_tensor = net_u(XYT_tensor, training=False)
        k_tensor = net_k(u_tensor, training=False)
        u_field = u_tensor.numpy().reshape(nx, ny)
        k_field = k_tensor.numpy().reshape(nx, ny)

        ax_u = axes_u[i]
        im_u = ax_u.contourf(X, Y, u_field, levels=100, cmap='YlOrBr', vmin=0, vmax=1)
        if i == 0: mappable_for_colorbar = im_u
        ax_u.set_title(f't = {t:.2f}')
        ax_u.set_xlabel('x')
        if i == 0: ax_u.set_ylabel('y')
        ax_u.axis("equal")

        ax_k = axes_k[i]
        # Use scatter for clarity
        ax_k.plot(u_field.flatten(), k_field.flatten(), 'o', markersize=1, label='Inferred k(u)', color='blue', alpha=0.1) 
        u_true_range = np.linspace(np.min(u_field), np.max(u_field), 100)
        k_true = get_ref_k(u_true_range)
        ax_k.plot(u_true_range, k_true, label='Reference k(u)', color='red', linewidth=2)
        ax_k.set_title(f't = {t:.2f}')
        ax_k.set_xlabel('u')
        ax_k.set_xlim(np.min(u_field), np.max(u_field))
        ax_k.set_ylim(0, 0.03)
        ax_k.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax_k.set_ylabel('k(u)')
        if i == len(time_slices) - 1: ax_k.legend()

    fig_u.subplots_adjust(right=0.85)
    cbar_ax_u = fig_u.add_axes([0.88, 0.15, 0.01, 0.7])
    if mappable_for_colorbar:
        fig_u.colorbar(mappable_for_colorbar, cax=cbar_ax_u, label='u(x,y,t)')
    fig_u.savefig(os.path.join(folder_name, f"temperature_epoch_{epoch:06d}.png"))
    plt.close(fig_u)

    fig_k.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_k.savefig(os.path.join(folder_name, f"conductivity_epoch_{epoch:06d}.png"))
    plt.close(fig_k)
    print(f"--- Saved plots for epoch {epoch} ---")

# ------------------------------
# Saving field function 
# ------------------------------

def save_field_data(net_u, net_k, folder_name="final_results_LBFGS"): # <-- Changed default name
    os.makedirs(folder_name, exist_ok=True)
    nx = ny = 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, 5)  
    records = []

    for ti in t:
        X, Y = np.meshgrid(x, y)
        T = ti * np.ones_like(X)
        XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
        XYT_tensor = tf.convert_to_tensor(XYT_grid)

        u_tensor = net_u(XYT_tensor, training=False)
        k_tensor = net_k(u_tensor, training=False)

        df_slice = pd.DataFrame({
            "x": XYT_grid[:,0],
            "y": XYT_grid[:,1],
            "t": XYT_grid[:,2],
            "u_pred": u_tensor.numpy().flatten(),
            "k_pred": k_tensor.numpy().flatten(),
        })
        records.append(df_slice)

    df_all = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(folder_name, "predicted_fields.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Saved field predictions to {csv_path}")


# ------------------------------
# L-BFGS-B Training setup
# ------------------------------

# L-BFGS-B does not use a Keras optimizer object
# We still use Checkpoints to save/restore the model weights
checkpoint_dir = './training_checkpoints_lbfgs' # Use a different directory
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_epoch = tf.Variable(1, dtype=tf.int64)
# Checkpoint now only tracks the models and the step number
ckpt = tf.train.Checkpoint(step=ckpt_epoch, net_u=net_u, net_k=net_k)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

results_folder = 'training_results_lbfgs'
loss_log_file = "training_loss_log_lbfgs.csv"

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")
    start_epoch = int(ckpt_epoch)
    try:
        loss_history_df = pd.read_csv(loss_log_file)
        loss_history = loss_history_df.values.tolist()
    except FileNotFoundError:
        loss_history = []
else:
    print("Initializing from scratch.")
    start_epoch = 1
    loss_history = []

# Logger class to handle callbacks from scipy.minimize
class LBFGSLogger:
    def __init__(self, start_iter=1, loss_history=[]):
        self.iter = start_iter
        self.loss_history = loss_history
        self.current_loss = 0.0
        self.loss_u = 0.0
        self.loss_k = 0.0

    def log_step(self, total_loss, loss_u, loss_k):
        """Called from inside the loss/grad function to store latest values."""
        self.current_loss = total_loss
        self.loss_u = loss_u
        self.loss_k = loss_k
    
    def callback(self, xk):
        """
        Called by scipy.minimize after each successful iteration.
        'xk' is the current parameter vector (which we don't need since weights are set).
        """
        self.loss_history.append([self.iter, self.current_loss, self.loss_u, self.loss_k])
        
        if self.iter % 200 == 0:
            print(f"L-BFGS Iter {self.iter:05d} — Total Loss: {self.current_loss:.4e}")
            ckpt_epoch.assign(self.iter)
            save_path = manager.save()
            print(f"Saved checkpoint for iter {self.iter} at {save_path}")
            df_loss = pd.DataFrame(self.loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
            df_loss.to_csv(loss_log_file, index=False)
        
        if self.iter % 2500 == 0:
            plot_and_save_fields(net_u, net_k, self.iter, results_folder)
        
        self.iter += 1

# ------------------------------
# Training loop (L-BFGS-B)
# ------------------------------

epochs = 25_000 # This will be 'maxiter' for L-BFGS
start_time = time.perf_counter()

print("Plotting initial random state (Epoch 0)...")
# This call BUILDS the models (net_u and net_k) if they weren't restored
plot_and_save_fields(net_u, net_k, 0, results_folder)

# --- START OF LBFGS HELPER FUNCTIONS ---
# Define these functions *after* the models are built by the plot call above.

# Get all trainable variables from both models
trainable_vars = net_u.trainable_variables + net_k.trainable_variables
# Store shapes and sizes for flattening/un-flattening
var_shapes = [v.shape for v in trainable_vars]
var_sizes = [tf.reduce_prod(s).numpy() for s in var_shapes]

def set_weights(flat_weights_np):
    """Sets model weights from a flat numpy array."""
    flat_weights_tf = tf.convert_to_tensor(flat_weights_np, dtype=tf.float32)
    idx = 0
    for var, size, shape in zip(trainable_vars, var_sizes, var_shapes):
        var.assign(tf.reshape(flat_weights_tf[idx : idx + size], shape))
        idx += size

def get_weights():
    """Gets model weights as a flat numpy array."""
    # This tf.concat will no longer get an empty list
    flat_weights = tf.concat([tf.reshape(v, [-1]) for v in trainable_vars], axis=0)
    return flat_weights.numpy()

@tf.function
def tf_loss_and_grads():
    """Computes loss and grads using tf.function for speed."""
    with tf.GradientTape() as tape:
        # L-BFGS uses the *full* dataset, not mini-batches
        total_loss, loss_u, loss_k = compute_loss(
            xyt_domain=xyt_dom,
            xyt_bc_dirichlet=xyt_bc_dirichlet, u_bc_dirichlet=u_bc_dirichlet,
            xyt_bc_neumann=xyt_bc_neumann,
            xyt_ic=xyt_ic, u_ic=u_ic,
            xyt_obs=XYT_obs, u_obs=u_obs, k_obs=k_obs, # <-- Full dataset
        )
    grads = tape.gradient(total_loss, trainable_vars)
    flat_grads = tf.concat([tf.reshape(g, [-1]) if g is not None else tf.zeros(s) for g, s in zip(grads, var_sizes)], axis=0)
    return total_loss, flat_grads, loss_u, loss_k

# Instantiate the logger
loss_logger = LBFGSLogger(start_iter=start_epoch, loss_history=loss_history)

def scipy_loss_and_grads(flat_weights_np):
    """
    Wrapper for SciPy.
    Sets weights, computes loss/grads, and returns them as float64.
    """
    # Set the model weights
    set_weights(flat_weights_np)
    
    # Compute loss and gradients
    total_loss, flat_grads, loss_u, loss_k = tf_loss_and_grads()
    
    # Log for callback
    total_loss_np = total_loss.numpy()
    loss_logger.log_step(total_loss_np, loss_u.numpy(), loss_k.numpy()) 
    
    # Convert to float64 for SciPy
    total_loss_f64 = total_loss_np.astype(np.float64)
    flat_grads_f64 = flat_grads.numpy().astype(np.float64)
    
    return total_loss_f64, flat_grads_f64

# --- END OF LBFGS HELPER FUNCTIONS ---


# Get initial weights
init_weights = get_weights() # This will now work

print("\n*** Starting L-BFGS-B training... ***")

# Run the optimizer
results = minimize(
    fun=scipy_loss_and_grads,  # Function to minimize (returns loss and grads)
    x0=init_weights,           # Initial guess
    method='L-BFGS-B',
    jac=True,                  # 'fun' returns the jacobian (gradient)
    callback=loss_logger.callback, # Function to call after each iteration
    options={
        'maxiter': epochs - start_epoch + 1,
        'disp': True,       # Print convergence messages
        'ftol': 1e-12,      # Tolerances
        'gtol': 1e-12
    }
)

end_time = time.perf_counter()
elapsed = end_time - start_time

# Set the final optimized weights
if results.success:
    print("Optimization terminated successfully.")
    set_weights(results.x)
else:
    print("Optimization failed.")
    # Weights are already set to the last successful step by the callback/scipy

final_loss = results.fun

print(f"\n*** L-BFGS-B training completed in {elapsed:.2f} seconds. ***")
print(f"Final L-BFGS-B loss: {final_loss:.4e}")

# Save final plots and data
plot_and_save_fields(net_u, net_k, loss_logger.iter, "final_results_LBFGS")
save_field_data(net_u, net_k, "final_results_LBFGS")

# Save final loss log
df_loss = pd.DataFrame(loss_logger.loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv(loss_log_file, index=False)

print("\n*** Script finished. ***")