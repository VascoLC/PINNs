import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

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

class NetS(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(10, activation="tanh") for _ in range(3)]
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
            for layer in self.hidden:
                x = layer(x)
            return self.out(x)
# ------------------------------
# Load measurement data
# ------------------------------

df = pd.read_csv("C:\PINNs_Git\PINNs\odil\examples\heat/2D/imposed.csv")
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
net_s = NetS()

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
            s = net_s(xyt)
        grads = tape1.gradient(u, xyt)
        du_dx, du_dy, du_dt = grads[:,0:1], grads[:,1:2], grads[:,2:3]
        flux_x = k * du_dx
        flux_y = k * du_dy
    flux_xx = tape2.gradient(flux_x, xyt)[:,0:1]
    flux_yy = tape2.gradient(flux_y, xyt)[:,1:2]
    return du_dt - (flux_xx + flux_yy) - s

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
        ax_k.plot(u_field.flatten(), k_field.flatten(), label='Inferred k(u)', color='blue')
        u_true_range = np.linspace(np.min(u_field), np.max(u_field), 100)
        k_true = get_ref_k(u_true_range)
        ax_k.plot(u_true_range, k_true, label='Reference k(u)', color='red')
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

def save_field_data(net_u, net_k, folder_name="final_results_ADAM"):
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

def plot_source_field(net_s, epoch, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    nx = ny = 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, len(time_slices), figsize=(18, 4), sharey=True)
    fig.suptitle('Inferred Source S(x,y,t)', fontsize=16)
    for i, t in enumerate(time_slices):
        T = t * np.ones_like(X)
        XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
        XYT_tensor = tf.convert_to_tensor(XYT_grid)
        S_pred = net_s(XYT_tensor, training=False).numpy().reshape(nx, ny)
        ax = axes[i]
        im = ax.contourf(X, Y, S_pred, levels=100, cmap='RdBu_r')
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('x')
        if i == 0: ax.set_ylabel('y')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='S(x,y,t)')
    fig.savefig(os.path.join(folder_name, f"source_epoch_{epoch:06d}.png"))
    plt.close(fig)



# ------------------------------
# Training loop
# ------------------------------

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_epoch = tf.Variable(1, dtype=tf.int64)
ckpt = tf.train.Checkpoint(step=ckpt_epoch, optimizer=optimizer, net_u=net_u, net_k=net_k)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

results_folder = 'training_results'

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")
    start_epoch = int(ckpt_epoch)
    try:
        loss_history_df = pd.read_csv("training_loss_log.csv")
        loss_history = loss_history_df.values.tolist()
    except FileNotFoundError:
        loss_history = []
else:
    print("Initializing from scratch.")
    start_epoch = 1
    loss_history = []

batch_size_obs = 500
num_obs = XYT_obs_np.shape[0]
num_batches = int(np.ceil(num_obs / batch_size_obs)) if num_obs > 0 else 0
epochs = 125_000
start_time = time.perf_counter()

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
            grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables+ net_s.trainable_variables)
            optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables+ net_s.trainable_variables))
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

    if epoch % 200 == 0:
        print(f"Epoch {epoch:05d} — Total Loss: {total_loss.numpy():.4e}")
        ckpt_epoch.assign(epoch)
        save_path = manager.save()
        print(f"Saved checkpoint for epoch {epoch} at {save_path}")
        df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
        df_loss.to_csv("training_loss_log.csv", index=False)
    if epoch % 200 == 0:
        plot_and_save_fields(net_u, net_k, epoch, results_folder)
        plot_source_field(net_s, epoch, results_folder)
        
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** ADAM training completed in {elapsed:.2f} seconds. ***")
print(f"Final ADAM loss: {total_loss.numpy():.4e}")

plot_and_save_fields(net_u, net_k, epochs, "final_results_ADAM")
save_field_data(net_u, net_k, "final_results_ADAM")

print("\n*** Script finished. ***")
