#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd

# =========================================================
# Fourier-feature Physics-Informed Neural Network (PINN)
# with NTK-based adaptive weighting for the wave equation
# =========================================================

# ------------------------------
# Neural Network Architecture
# ------------------------------
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, sigma_x=[1], sigma_t=[1,10]):
        super().__init__()
        self.Bx = tf.constant(sigma_x, dtype=tf.float32)
        self.Bt = tf.constant(sigma_t, dtype=tf.float32)

    def call(self, x):
        x_spatial = x[:, 0:1]
        t_temporal = x[:, 1:2]
        fx = tf.concat([tf.sin(2*np.pi*x_spatial*self.Bx),
                        tf.cos(2*np.pi*x_spatial*self.Bx)], axis=1)
        ft = tf.concat([tf.sin(2*np.pi*t_temporal*self.Bt),
                        tf.cos(2*np.pi*t_temporal*self.Bt)], axis=1)
        return tf.concat([x, fx, ft], axis=1)


class FourierPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fourier = FourierLayer(sigma_x=[1], sigma_t=[1, 10])
        self.hidden = [tf.keras.layers.Dense(100, activation="tanh") for _ in range(3)]
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fourier(x)
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

# ------------------------------
# Exact solution and derivatives
# ------------------------------
def U_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:]
    u_exact = np.zeros_like(x_)
    for k in range(1, 6):
        u_exact += 1/10 * (np.cos((x_-t_+0.5)*np.pi*k) + np.cos((x_+t_+0.5)*np.pi*k))
    return u_exact

def Ut_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:2]
    ut = np.zeros_like(x_)
    for k in range(1, 6):
        ut += 1/10 * np.pi * k * (
            np.sin((x_ - t_ + 0.5)*np.pi*k) - np.sin((x_ + t_ + 0.5)*np.pi*k)
        )
    return ut

# ------------------------------
# Generate training data
# ------------------------------
# Initial condition
x_ic = np.linspace(-1, 1, 400)[:, None]
t_ic = np.zeros_like(x_ic)
init_xt = np.hstack([x_ic, t_ic])
init_u = U_exact(init_xt)
init_ut = Ut_exact(init_xt)
xt_ic = tf.convert_to_tensor(init_xt, dtype=tf.float32)
u_ic = tf.convert_to_tensor(init_u, dtype=tf.float32)
ut_ic = tf.convert_to_tensor(init_ut, dtype=tf.float32)

# Left and right boundary conditions
t_bc = np.linspace(0, 1, 400)[:, None]
x_bc_left = -np.ones_like(t_bc)
x_bc_right = np.ones_like(t_bc)
l_bc_np = np.hstack([x_bc_left, t_bc])
r_bc_np = np.hstack([x_bc_right, t_bc])
l_bc = tf.convert_to_tensor(l_bc_np, dtype=tf.float32)
r_bc = tf.convert_to_tensor(r_bc_np, dtype=tf.float32)
u_bc_left_tf = tf.convert_to_tensor(U_exact(l_bc_np), dtype=tf.float32)
u_bc_right_tf = tf.convert_to_tensor(U_exact(r_bc_np), dtype=tf.float32)

# Observation (data) points
n_obs = 500
x_obs = np.random.uniform(-1, 1, (n_obs, 1))
t_obs = np.random.uniform(0, 1, (n_obs, 1))
xt_obs_np = np.hstack([x_obs, t_obs])
xt_obs = tf.convert_to_tensor(xt_obs_np, dtype=tf.float32)
u_obs = tf.convert_to_tensor(U_exact(xt_obs_np), dtype=tf.float32)

# Domain collocation points
n_dom = 4096
x_dom = np.random.uniform(-1, 1, (n_dom, 1))
t_dom = np.random.uniform(0, 1, (n_dom, 1))
xt_dom_np = np.hstack([x_dom, t_dom])
xt_dom = tf.convert_to_tensor(xt_dom_np, dtype=tf.float32)

# ------------------------------
# PDE residual (wave equation)
# ------------------------------
def pde_residual(model, x):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            u = model(x)
        du_dx = t1.gradient(u, x)[:, 0:1]
        du_dt = t1.gradient(u, x)[:, 1:2]
    du_xx = t2.gradient(du_dx, x)[:, 0:1]
    du_tt = t2.gradient(du_dt, x)[:, 1:2]
    return du_tt - du_xx

# ------------------------------
# Model instance and optimizer
# ------------------------------
net_u = FourierPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# ------------------------------
# Loss components
# ------------------------------
def compute_loss(xt_domain, l_bc, r_bc, xt_ic, u_ic, ut_ic, u_bc_l, u_bc_r, xt_obs, u_obs):
    loss_pde = tf.reduce_mean(tf.square(pde_residual(net_u, xt_domain)))
    loss_ic1 = tf.reduce_mean(tf.square(net_u(xt_ic) - u_ic))

    with tf.GradientTape() as tape1:
        tape1.watch(xt_ic)
        u_ic_pred = net_u(xt_ic)
    ut_ic_pred = tape1.gradient(u_ic_pred, xt_ic)[:, 1:]
    loss_ic2 = tf.reduce_mean(tf.square(ut_ic_pred - ut_ic))

    loss_lbc = tf.reduce_mean(tf.square(net_u(l_bc) - u_bc_l))
    loss_rbc = tf.reduce_mean(tf.square(net_u(r_bc) - u_bc_r))
    loss_obs = tf.reduce_mean(tf.square(net_u(xt_obs) - u_obs))

    total_loss = loss_pde + loss_ic1 + loss_ic2 + loss_lbc + loss_rbc + loss_obs
    return total_loss, {
        "pde": loss_pde, "ic1": loss_ic1, "ic2": loss_ic2,
        "lbc": loss_lbc, "rbc": loss_rbc, "data": loss_obs
    }

# ------------------------------
# Initialize NTK-based weights
# ------------------------------
lambda_pde = tf.Variable(1.0, trainable=False)
lambda_ic1 = tf.Variable(1.0, trainable=False)
lambda_ic2 = tf.Variable(1.0, trainable=False)
lambda_lbc = tf.Variable(1.0, trainable=False)
lambda_rbc = tf.Variable(1.0, trainable=False)
lambda_obs = tf.Variable(1.0, trainable=False)
lambdas = [lambda_pde, lambda_ic1, lambda_ic2, lambda_lbc, lambda_rbc, lambda_obs]

# ------------------------------
# Trace-of-NTK computation
# ------------------------------
def trace_ntk(model, x_in, loss_term_fn):
    """Approximate Tr(K) = sum of squared gradients of a given loss term."""
    with tf.GradientTape() as tape:
        # Forward pass through loss function
        loss_term = tf.reduce_mean(loss_term_fn(x_in))
    grads = tape.gradient(loss_term, model.trainable_variables)

    # Filter out None grads safely
    grads = [g for g in grads if g is not None]
    if len(grads) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    # Compute sum of squared norms (trace approximation)
    squared_norms = [tf.reduce_sum(tf.square(g)) for g in grads]
    return tf.add_n(squared_norms)


# ------------------------------
# Training Loop
# ------------------------------
epochs = 40_000
loss_history = []
update_period = 1000  

# Grid for visualization
nx, nt = 200, 200
x_lin = np.linspace(-1, 1, nx)
t_lin = np.linspace(0, 1, nt)
X, T = np.meshgrid(x_lin, t_lin)
XT_grid = np.hstack([X.flatten()[:, None], T.flatten()[:, None]]).astype(np.float32)
XT_tensor = tf.convert_to_tensor(XT_grid, dtype=tf.float32)

for epoch in range(epochs):
    # Compute loss and gradients
    with tf.GradientTape() as tape:
        total_loss, losses = compute_loss(
            xt_dom, l_bc, r_bc, xt_ic, u_ic, ut_ic,
            u_bc_left_tf, u_bc_right_tf, xt_obs, u_obs
        )
        weighted_loss = (
            lambda_pde * losses["pde"] +
            lambda_ic1 * losses["ic1"] +
            lambda_ic2 * losses["ic2"] +
            lambda_lbc * losses["lbc"] +
            lambda_rbc * losses["rbc"] +
            lambda_obs * losses["data"]
        )
    grads = tape.gradient(weighted_loss, net_u.trainable_variables)
    optimizer.apply_gradients(zip(grads, net_u.trainable_variables))

    # ---- NTK-based adaptive λ update ----
    if epoch % update_period == 0:
        Tr_Ku = trace_ntk(net_u, xt_ic, lambda x: net_u(x) - u_ic)
        with tf.GradientTape() as tgrad:
            tgrad.watch(xt_ic)
            u_ic_pred = net_u(xt_ic)
        ut_ic_pred = tgrad.gradient(u_ic_pred, xt_ic)[:, 1:]
        Tr_Kut = trace_ntk(net_u, xt_ic, lambda x: ut_ic_pred - ut_ic)
        Tr_Kr = trace_ntk(net_u, xt_dom, lambda x: pde_residual(net_u, x))

        Tr_sum = Tr_Ku + Tr_Kut + Tr_Kr
        lambda_ic1.assign(tf.clip_by_value(Tr_sum / Tr_Ku, 1e-2, 1e3))
        lambda_ic2.assign(tf.clip_by_value(Tr_sum / Tr_Kut, 1e-2, 1e3))
        lambda_pde.assign(tf.clip_by_value(Tr_sum / Tr_Kr, 1e-2, 1e3))

        print(f"[λ update] λ_u={lambda_ic1.numpy():.3e}, λ_ut={lambda_ic2.numpy():.3e}, λ_r={lambda_pde.numpy():.3e}")

    # ---- Record and save ----
    weighted_total_loss = (
        lambda_pde * losses["pde"] +
        lambda_ic1 * losses["ic1"] +
        lambda_ic2 * losses["ic2"]
    )

    loss_history.append([
        epoch,
        float(weighted_total_loss),
        float(losses["pde"]),
        float(losses["ic1"]),
        float(losses["ic2"]),
        float(losses["lbc"]),
        float(losses["rbc"]),
        float(losses["data"]),
    ])

    if epoch % 500 == 0:
        u_pred = net_u(XT_tensor).numpy().flatten()
        df_field = pd.DataFrame({
            "x": XT_grid[:, 0],
            "t": XT_grid[:, 1],
            "u_pred": u_pred
        })
        filename = f"field_epoch_{epoch:06d}.csv"
        df_field.to_csv(filename, index=False)
        print(f"Saved predicted field to {filename}")
        print(
            f"Epoch {epoch:05d} | Weighted Loss: {weighted_total_loss.numpy():.3e} | "
            f"PDE: {losses['pde'].numpy():.3e} | Data: {losses['data'].numpy():.3e}"
        )

# ------------------------------
# Save training history
# ------------------------------
columns = ["epoch", "total", "pde", "ic1", "ic2", "lbc", "rbc", "obs_u"]
loss_df = pd.DataFrame(loss_history, columns=columns)
loss_df.to_csv("loss_history.csv", index=False)
print("Saved loss history to loss_history.csv")
