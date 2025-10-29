import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

# ------------------------------
# Force TF to use float32
# ------------------------------
tf.keras.backend.set_floatx('float32')

# ------------------------------
# Reference conductivity
# ------------------------------
def get_ref_k(u, mod=np):
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))

# ------------------------------
# Neural Networks
# ------------------------------
class NetU(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(32, activation="tanh") for _ in range(4)]
        self.out = tf.keras.layers.Dense(1, activation=None)
    def call(self, x):
        for h in self.hidden:
            x = h(x)
        return self.out(x)

class NetK(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(5, activation="tanh") for _ in range(2)]
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")
    def call(self, u):
        # u is in [0,1]; normalize to ~[-1,1] to help the small network
        u_norm = 2.0 * (u - 0.5)
        x = u_norm
        for h in self.hidden:
            x = h(x)
        z = self.out(x)
        return 0.1 * z  # scale to k_max=0.1

# ------------------------------
# Load imposed (x,t,u) data
# ------------------------------
df = pd.read_csv(r"C:\PINNs_Git\PINNs\odil\examples\heat\1D\out_heat_inverse_1D_Imposed_solution\imposed_data_with_u.csv")
XT_obs_np = df[["x", "t"]].values.astype(np.float32)
u_obs_np  = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np  = get_ref_k(u_obs_np).astype(np.float32)

XT_obs = tf.convert_to_tensor(XT_obs_np, dtype=tf.float32)
u_obs  = tf.convert_to_tensor(u_obs_np,  dtype=tf.float32)
k_obs  = tf.convert_to_tensor(k_obs_np,  dtype=tf.float32)

# ------------------------------
# IC and BC (float32)
# ------------------------------
n_ic = 400
x_ic = np.random.rand(n_ic, 1).astype(np.float32)
t_ic = np.zeros((n_ic, 1), dtype=np.float32)
u_ic_np = (np.exp(-50*(x_ic-0.5)**2) - np.exp(-50*0.5**2)).astype(np.float32)

xt_ic = tf.convert_to_tensor(np.hstack([x_ic, t_ic]).astype(np.float32))
u_ic  = tf.convert_to_tensor(u_ic_np, dtype=tf.float32)

n_bc = 400
t_bc = np.random.rand(n_bc, 1).astype(np.float32)
edges = np.random.choice(["left", "right"], size=n_bc)
x_bc = np.random.rand(n_bc, 1).astype(np.float32)
x_bc[edges == "left"]  = 0.0
x_bc[edges == "right"] = 1.0

xt_bc = tf.convert_to_tensor(np.hstack([x_bc, t_bc]).astype(np.float32))
u_bc  = tf.zeros((n_bc, 1), dtype=tf.float32)

# ------------------------------
# Domain samples for PDE residual
# ------------------------------
n_dom = 4096
x_dom = np.random.rand(n_dom, 1).astype(np.float32)
t_dom = np.random.rand(n_dom, 1).astype(np.float32)
xt_dom = tf.convert_to_tensor(np.hstack([x_dom, t_dom]).astype(np.float32))

# ------------------------------
# Models (built in float32)
# ------------------------------
net_u = NetU()
net_k = NetK()
_ = net_u(tf.zeros((1, 2), dtype=tf.float32))
_ = net_k(tf.zeros((1, 1), dtype=tf.float32))

# ------------------------------
# PDE residual
# ------------------------------
def compute_pde_residual(xt):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xt)
            u = net_u(xt)         # (N,1)
            k = net_k(u)          # (N,1)
        grads = tape1.gradient(u, xt)  # (N,2): [du/dx, du/dt]
        du_dx, du_dt = grads[:, 0:1], grads[:, 1:2]
        flux_x = k * du_dx
    flux_xx = tape2.gradient(flux_x, xt)[:, 0:1]  # d/dx(k du/dx)
    return du_dt - flux_xx

# ------------------------------
# Loss
# ------------------------------
def compute_loss(xt_domain, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs, k_obs):
    res = compute_pde_residual(xt_domain)
    loss_pde = tf.reduce_mean(tf.square(res))

    u_pred_ic = net_u(xt_ic)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic))

    u_pred_bc = net_u(xt_bc)
    loss_bc = tf.reduce_mean(tf.square(u_pred_bc - u_bc))

    u_pred_obs = net_u(xt_obs)
    loss_u = tf.reduce_mean(tf.square(u_pred_obs - u_obs))

    k_pred_obs = net_k(u_pred_obs)
    loss_k = tf.reduce_mean(tf.square(k_pred_obs - k_obs))

    total_loss = loss_pde + loss_ic + loss_bc + loss_u + loss_k
    return total_loss, loss_pde, loss_ic, loss_bc, loss_u, loss_k

# ------------------------------
# Flatten / unflatten (float32 inside TF)
# ------------------------------
def get_weights():
    vars_ = net_u.trainable_variables + net_k.trainable_variables
    flat_list = [tf.reshape(v, [-1]) for v in vars_]
    return tf.concat(flat_list, axis=0)  # float32

def set_weights(flat_vector):
    vars_ = net_u.trainable_variables + net_k.trainable_variables
    idx = 0
    for v in vars_:
        size = int(np.prod(v.shape))
        new_val = tf.reshape(flat_vector[idx: idx + size], v.shape)
        v.assign(new_val)  # both float32
        idx += size

# ------------------------------
# L-BFGS-B wrapper
# ------------------------------
loss_log = []  # [total, pde, ic, bc, u, k]

def loss_and_grad(flat_weights_np):
    # SciPy provides float64 -> cast to float32 before using in TF
    flat_weights = tf.convert_to_tensor(flat_weights_np, dtype=tf.float32)
    with tf.GradientTape() as tape:
        set_weights(flat_weights)
        total, lpde, lic, lbc, lu, lk = compute_loss(
            xt_domain=xt_dom,
            xt_bc=xt_bc, u_bc=u_bc,
            xt_ic=xt_ic, u_ic=u_ic,
            xt_obs=XT_obs, u_obs=u_obs, k_obs=k_obs,
        )
    vars_ = net_u.trainable_variables + net_k.trainable_variables
    grads = tape.gradient(total, vars_)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, vars_)]
    grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)  # float32

    # Log (as python floats)
    loss_log.append([
        float(total.numpy()),
        float(lpde.numpy()),
        float(lic.numpy()),
        float(lbc.numpy()),
        float(lu.numpy()),
        float(lk.numpy()),
    ])

    # SciPy expects float64 numpy arrays
    return float(total.numpy()), grads_flat.numpy().astype(np.float64)

def callback(_x):
    if loss_log:
        print(f"Iter {len(loss_log):05d}: total loss = {loss_log[-1][0]:.6e}")

# ------------------------------
# Run optimization
# ------------------------------
print("Starting L-BFGS-B optimization...")
start = time.perf_counter()

x0 = get_weights().numpy().astype(np.float64)  # SciPy wants float64
x_opt, f_opt, info = fmin_l_bfgs_b(
    func=loss_and_grad,
    x0=x0,
    maxiter=50000,
    m=50,
    factr=1e7,
    pgtol=1e-8,
    maxfun=50000,
    callback=callback,
)

elapsed = time.perf_counter() - start
print(f"\nOptimization completed in {elapsed:.2f} s")
print(f"Final loss: {f_opt:.4e}")
print(f"Exit flag: {info['warnflag']} — {info['task']}")

# Set the final weights (cast back to float32)
set_weights(tf.convert_to_tensor(x_opt.astype(np.float32)))

# ------------------------------
# Save logs and final field
# ------------------------------
loss_df = pd.DataFrame(loss_log, columns=["total","pde","ic","bc","u","k"])
loss_df.insert(0, "iteration", np.arange(1, len(loss_df)+1))
loss_df.to_csv("training_loss_log.csv", index=False)
print("✅ Saved training_loss_log.csv")

Nx, Nt = 200, 200
x = np.linspace(0, 1, Nx, dtype=np.float32)
t = np.linspace(0, 1, Nt, dtype=np.float32)
X, T = np.meshgrid(x, t)
XT_grid = np.hstack([X.flatten()[:,None], T.flatten()[:,None]]).astype(np.float32)

u_pred = net_u(tf.convert_to_tensor(XT_grid)).numpy().reshape(Nt, Nx)
k_pred = net_k(net_u(tf.convert_to_tensor(XT_grid))).numpy().reshape(Nt, Nx)

field_df = pd.DataFrame({
    "x": X.flatten().astype(np.float32),
    "t": T.flatten().astype(np.float32),
    "u_pred": u_pred.flatten().astype(np.float32),
    "k_pred": k_pred.flatten().astype(np.float32),
})
field_df.to_csv("Heat_Inverse_1D_solution_LBFGSB.csv", index=False)
print("✅ Saved Heat_Inverse_1D_solution_LBFGSB.csv")

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(7, 15))
cf = plt.contourf(X, T, u_pred, levels=100, cmap="viridis")
plt.colorbar(cf, label="u(x,t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN solution u(x,t) after L-BFGS-B")
plt.tight_layout()
plt.show()
