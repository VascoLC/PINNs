import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ------------------------------
# Transfer Learning
# ------------------------------

def pack_variables(var_list):
    """Flatten a list of Tensors to a 1D tf.Tensor."""
    flat_vars = [tf.reshape(v, [-1]) for v in var_list]
    return tf.concat(flat_vars, axis=0)

def unpack_variables(theta, var_list_template):
    """Assign a flat theta vector back into the shapes of var_list_template."""
    idx = 0
    assigns = []
    for v in var_list_template:
        size = tf.size(v)
        new_val = tf.reshape(theta[idx:idx+size], tf.shape(v))
        assigns.append(v.assign(new_val))
        idx += size
    tf.debugging.assert_equal(idx, tf.size(theta))
    return assigns

# --- Improved make_loss_and_grad_fn ---
def make_loss_and_grad_fn():
    var_list = net_u.trainable_variables + net_k.trainable_variables

    @tf.function
    def value_and_grad(theta):
        # Assign the flat parameter vector to the model variables
        unpack_variables(theta, var_list)

        # Compute the loss using a GradientTape
        with tf.GradientTape() as tape:
            # The tape automatically watches the trainable variables
            total_loss, _, _ = compute_loss(
                xyt_domain=xyt_dom,
                xyt_bc_dirichlet=xyt_bc_dirichlet, u_bc_dirichlet=u_bc_dirichlet,
                xyt_bc_neumann=xyt_bc_neumann,
                xyt_ic=xyt_ic, u_ic=u_ic,
                xyt_obs=XYT_obs, u_obs=u_obs, k_obs=k_obs,
            )
        
        # Calculate gradients of the loss with respect to the variables
        grads = tape.gradient(total_loss, var_list)
        
        # Pack the gradients back into a flat vector
        grad_theta = pack_variables(grads)
        return total_loss, grad_theta

    return var_list, value_and_grad

lbfgs_log = [] 

def run_lbfgs_tfp(max_iterations=500, tolerance=1e-9, log_path="lbfgs_log_lbfgs.csv"):
    
    var_list, value_and_grad = make_loss_and_grad_fn()
    theta0 = pack_variables(var_list)

    # --- Step 1: Set up for logging ---
    lbfgs_log = []
    iteration_counter = tf.Variable(0, dtype=tf.int32)

    # MODIFICATION: The log_step function now accepts all three loss values.
    def log_step(total_loss, loss_u, loss_k):
        iteration = iteration_counter.numpy()
        if iteration % 20 == 0:
            print(f"L-BFGS Iteration: {iteration:04d}, Total Loss: {total_loss.numpy():.4e}, "
                  f"Loss_u: {loss_u.numpy():.4e}, Loss_k: {loss_k.numpy():.4e}")
        
        # Append all three loss values to our log list.
        lbfgs_log.append([iteration, total_loss.numpy(), loss_u.numpy(), loss_k.numpy()])
        return total_loss # tf.py_function must return something

    # --- Step 2: Create the target function for the optimizer ---
    def tfp_target(theta):
    # Define an inner function to compute loss and gradients
        @tf.function
        def decorated_value_and_grad(t):
            unpack_variables(t, var_list)
            with tf.GradientTape() as tape:
                # Capture all three loss components
                total, u, k = compute_loss(
                    xyt_domain=xyt_dom,
                    xyt_bc_dirichlet=xyt_bc_dirichlet, u_bc_dirichlet=u_bc_dirichlet,
                    xyt_bc_neumann=xyt_bc_neumann,
                    xyt_ic=xyt_ic, u_ic=u_ic,
                    xyt_obs=XYT_obs, u_obs=u_obs, k_obs=k_obs,
                )
            grads = tape.gradient(total, var_list)

            # --- 🔍 DEBUG BLOCK: check for missing gradients ---
            for g, v in zip(grads, var_list):
                if g is None:
                    print(f"⚠️ No gradient for variable: {getattr(v, 'name', 'unnamed variable')}")
            # ----------------------------------------------------

            flat_grads = pack_variables(grads)
            return total, u, k, flat_grads

        # Run the inner function
        total_loss, loss_u, loss_k, grad = decorated_value_and_grad(theta)

        # --- Step 3: Inject the logging operation ---
        iteration_counter.assign_add(1)

        # Pass losses to the Python logger
        logged_total_loss = tf.py_function(
            log_step,
            inp=[total_loss, loss_u, loss_k],
            Tout=tf.float32,
        )

        # Return the original total_loss and gradient to the optimizer
        return logged_total_loss, grad

    
    print("--- Starting Phase 2: L-BFGS Optimization ---")
    results = tfp.optimizer.lbfgs_minimize(
        tfp_target,
        initial_position=theta0,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    unpack_variables(results.position, var_list)
    
    # --- Step 4: Save the detailed log data to a CSV file ---
    if lbfgs_log:
        # MODIFICATION: Update column names for the new data.
        pd.DataFrame(
            lbfgs_log, 
            columns=["iter", "total_loss", "loss_u", "loss_k"]
        ).to_csv(log_path, index=False)
        print(f"Saved detailed L-BFGS progress to {log_path}")

    print(f"\n[L-BFGS/TFP] Converged: {results.converged.numpy()} "
          f"| Iterations: {results.num_iterations.numpy()} "
          f"| Final loss: {results.objective_value.numpy():.4e}")
# ------------------------------
# Reference Conductivity
# ------------------------------

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
# 2) Load measurement data
# ------------------------------

df = pd.read_csv("C:/PINNs_Git/PINNs/odil/examples/heat/2D/First_Step/out_heat_inverse_2D_32_IMPOSEDPOINTS/imposed_data_with_u_2D.csv")
XYT_obs_np = df[["x", "y", "t"]].values.astype(np.float32)
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np   = get_ref_k(u_obs_np)
XYT_obs = tf.convert_to_tensor(XYT_obs_np)
u_obs   = tf.convert_to_tensor(u_obs_np)
k_obs   = tf.convert_to_tensor(k_obs_np)

# ------------------------------
# 3) Generate IC and BC samples
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
    loss_pde = tf.reduce_mean(tf.square(compute_pde_residual(xyt_domain)))
    loss_ic = tf.reduce_mean(tf.square(net_u(xyt_ic) - u_ic))
    loss_bc_dirichlet = tf.reduce_mean(tf.square(net_u(xyt_bc_dirichlet) - u_bc_dirichlet))
    with tf.GradientTape() as tape_neumann:
        tape_neumann.watch(xyt_bc_neumann)
        u_pred_bc_neumann = net_u(xyt_bc_neumann)
    du_dy_bc = tape_neumann.gradient(u_pred_bc_neumann, xyt_bc_neumann)[:, 1:2]
    loss_bc_neumann = tf.reduce_mean(tf.square(du_dy_bc))
    
    if xyt_obs.shape[0] > 0:
        u_pred_obs = net_u(xyt_obs)
        loss_u = tf.reduce_mean(tf.square(u_pred_obs - u_obs))
        k_pred_obs = net_k(u_pred_obs)
        loss_k = tf.reduce_mean(tf.square(k_pred_obs - k_obs))
    else:
        loss_u = tf.constant(0.0, dtype=tf.float32)
        loss_k = tf.constant(0.0, dtype=tf.float32)

    total_loss = (loss_pde + loss_ic + loss_bc_dirichlet + loss_bc_neumann + loss_u + 100*loss_k)
    return total_loss, loss_u, loss_k

# ------------------------------
# Plotting function 
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

        if i == 0:
            mappable_for_colorbar = im_u
        ax_u.set_title(f't = {t:.2f}')
        ax_u.set_xlabel('x')
        if i == 0:
            ax_u.set_ylabel('y')
        ax_u.axis("equal")

        ax_k = axes_k[i]
        ax_k.plot(u_field.flatten(), k_field.flatten(), label='Inferred k(u)', color='blue')

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

    fig_u.subplots_adjust(right=0.85)
    cbar_ax_u = fig_u.add_axes([0.88, 0.15, 0.01, 0.7])
    if mappable_for_colorbar:
        fig_u.colorbar(mappable_for_colorbar, cax=cbar_ax_u, label='u(x,y,t)')
    u_filename = os.path.join(folder_name, f"temperature_epoch_{str(epoch).zfill(6)}.png")
    fig_u.savefig(u_filename)
    plt.close(fig_u)

    k_filename = os.path.join(folder_name, f"conductivity_epoch_{str(epoch).zfill(6)}.png")
    fig_k.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_k.savefig(k_filename)
    plt.close(fig_k)
    
    print(f"--- Saved plots for epoch {epoch} ---")

# ------------------------------
# 8) Training loop (Adam)
# ------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# --- Checkpoint Setup ---
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_epoch = tf.Variable(1, dtype=tf.int64)
ckpt = tf.train.Checkpoint(step=ckpt_epoch, optimizer=optimizer, net_u=net_u, net_k=net_k)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

# --- MODIFIED: Results Directory ---
results_folder = 'training_results_FBatch'

# Restore from the latest checkpoint
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")
    start_epoch = int(ckpt_epoch)
    try:
        loss_history_df = pd.read_csv("training_loss_log_FBatch.csv")
        loss_history = loss_history_df.values.tolist()
        print(f"Loaded {len(loss_history)} previous log entries.")
    except FileNotFoundError:
        loss_history = []
        print("Starting new training log.")
else:
    print("Initializing from scratch.")
    start_epoch = 1
    loss_history = []

batch_size_obs = 500
num_obs = XYT_obs_np.shape[0]
num_batches = int(np.ceil(num_obs / batch_size_obs)) if num_obs > 0 else 0
epochs = 125_000
start_time = time.perf_counter()

# --- MODIFIED: Plot initial random state before training ---
print("Plotting initial random state (Epoch 0)...")
plot_and_save_fields(net_u, net_k, 0, results_folder)
total_loss = tf.constant(0.0)
for epoch in range(start_epoch, epochs + 1):
    if num_batches > 0:
        indices = np.random.permutation(num_obs)
        for batch in range(num_batches):
            batch_idx = indices[batch * batch_size_obs : (batch + 1) * batch_size_obs]
            XYT_obs_mb = tf.convert_to_tensor(XYT_obs_np[batch_idx])
            u_obs_mb   = tf.convert_to_tensor(u_obs_np[batch_idx])
            k_obs_mb   = tf.convert_to_tensor(k_obs_np[batch_idx])
            
            with tf.GradientTape() as tape:
                total_loss, loss_u, loss_k = compute_loss(
                    xyt_domain=xyt_dom,
                    xyt_bc_dirichlet=xyt_bc_dirichlet, u_bc_dirichlet=u_bc_dirichlet,
                    xyt_bc_neumann=xyt_bc_neumann,
                    xyt_ic=xyt_ic, u_ic=u_ic,
                    xyt_obs=XYT_obs_mb, u_obs=u_obs_mb, k_obs=k_obs_mb,
                )
            grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables)
            optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables))
    else: 
        with tf.GradientTape() as tape:
            total_loss, loss_u, loss_k = compute_loss(
                xyt_domain=xyt_dom,
                xyt_bc_dirichlet=xyt_bc_dirichlet, u_bc_dirichlet=u_bc_dirichlet,
                xyt_bc_neumann=xyt_bc_neumann,
                xyt_ic=xyt_ic, u_ic=u_ic,
                xyt_obs=XYT_obs, u_obs=u_obs, k_obs=k_obs,
            )
        grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables)
        optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables))
    
    loss_history.append([epoch, total_loss.numpy(), loss_u.numpy(), loss_k.numpy()])

    if epoch % 500 == 0:
        print(f"Epoch {epoch:05d} — Total Loss: {total_loss.numpy():.4e}")
        ckpt_epoch.assign(epoch)
        save_path = manager.save()
        print(f"Saved checkpoint for epoch {epoch} at {save_path}")
        df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
        df_loss.to_csv("training_loss_log_FBatch.csv", index=False)
        print("Saved training_loss_log.csv")
    if epoch % 2500 == 0:
        plot_and_save_fields(net_u, net_k, epoch, results_folder)
        
# ... (Keep the ADAM training loop exactly as it is) ...

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** ADAM training completed in {elapsed:.2f} seconds. ***")
print(f"Final ADAM loss: {total_loss.numpy():.4e}")

# --- Plot results after ADAM ---
print("\nGenerating plots after ADAM optimization...")
plot_and_save_fields(net_u, net_k, epochs, "final_results_ADAM")


# ------------------------------------
# 9) Phase 2: L-BFGS Optimization
# ------------------------------------
print("\n--- Starting Phase 2: L-BFGS Optimization ---")
run_lbfgs_tfp(max_iterations=200, tolerance=1e-9, log_path="training_loss_log_lbfgs.csv")


# ------------------------------------
# 10) Final Actions After All Training
# ------------------------------------
print("\nGenerating final high-resolution plots after L-BFGS...")
plot_and_save_fields(net_u, net_k, epochs, "final_results_LBFGS")

# --- Save the final model state ---
final_checkpoint_path = manager.save()
print(f"Saved final model checkpoint at {final_checkpoint_path}")

# --- Combine ADAM and L-BFGS logs ---
try:
    adam_log_df = pd.read_csv("training_loss_log_FBatch.csv")
    lbfgs_log_df = pd.read_csv("training_loss_log_lbfgs.csv")
    # Give L-BFGS iterations a higher "epoch" number to show sequence
    lbfgs_log_df['epoch'] = lbfgs_log_df['iter'] + adam_log_df['epoch'].max()
    
    # Combine and save
    full_log_df = pd.concat([adam_log_df[['epoch', 'total_loss']], lbfgs_log_df[['epoch', 'total_loss']].rename(columns={'loss': 'total_loss'})])
    full_log_df.to_csv("training_loss_log_FULL.csv", index=False)
    print("Saved combined training log to training_loss_log_FULL.csv")
except FileNotFoundError:
    print("Could not combine logs. Ensure log files exist.")

print("\n*** Script finished. ***")