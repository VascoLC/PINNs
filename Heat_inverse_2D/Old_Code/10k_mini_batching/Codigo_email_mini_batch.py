import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ------------------------------
# 1) Define the two neural nets
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
# 3) Load measurement data at t
# ------------------------------

# --- Load all your sampled measurements (at t=0.2,0.4,0.6,0.8,1.0) ---
df = pd.read_csv("FD_train_data.csv")

# If you want to use *all* of them:
XYT_obs_np = df[["x", "y", "t"]].values.astype(np.float32)   # (N, 3)
u_obs_np   = df[["u"]].values.reshape(-1, 1).astype(np.float32)  # (N, 1)
k_obs_np   = df[["k"]].values.reshape(-1, 1).astype(np.float32)  # (N, 1)

# Convert to tensors
XYT_obs = tf.convert_to_tensor(XYT_obs_np)
u_obs   = tf.convert_to_tensor(u_obs_np)
k_obs   = tf.convert_to_tensor(k_obs_np)

#u_max = np.max(u_obs_np)

# ------------------------------
# 4) Generate IC and BC samples
# ------------------------------

# 4.1) Initial condition: u(x,y,0) = 16 x (1-x) y (1-y)
n_ic = 400
x_ic = np.random.rand(n_ic,1)
y_ic = np.random.rand(n_ic,1)
t_ic = np.zeros((n_ic,1), dtype=np.float32)
xyt_ic_np = np.hstack([x_ic, y_ic, t_ic]).astype(np.float32)
u_ic_np   = (16 * x_ic * (1 - x_ic) * y_ic * (1 - y_ic)).astype(np.float32)

xyt_ic = tf.convert_to_tensor(xyt_ic_np)
u_ic   = tf.convert_to_tensor(u_ic_np)

# 4.2) Boundary condition: u = 0 on ∂Ω for t ∈ [0,1]
n_bc = 400
t_bc = np.random.rand(n_bc,1).astype(np.float32)
edges = np.random.choice(["left","right","bottom","top"], size=n_bc)

x_bc = np.random.rand(n_bc,1)
y_bc = np.random.rand(n_bc,1)
x_bc[edges=="left"]   = 0.0
x_bc[edges=="right"]  = 1.0
y_bc[edges=="bottom"] = 0.0
y_bc[edges=="top"]    = 1.0

xyt_bc_np = np.hstack([x_bc, y_bc, t_bc]).astype(np.float32)
u_bc_np   = np.zeros((n_bc,1), dtype=np.float32)

xyt_bc = tf.convert_to_tensor(xyt_bc_np)
u_bc   = tf.convert_to_tensor(u_bc_np)


# ------------------------------
# 5) Domain samples for PDE residual
# ------------------------------

n_dom = 4096
x_dom = np.random.rand(n_dom,1)
y_dom = np.random.rand(n_dom,1)
t_dom = np.random.rand(n_dom,1)
xyt_dom_np = np.hstack([x_dom, y_dom, t_dom]).astype(np.float32)
xyt_dom = tf.convert_to_tensor(xyt_dom_np)


# ------------------------------
# 6) Instantiate models
# ------------------------------

net_u = NetU()
net_k = NetK()


# ------------------------------
# 7) PDE residual function
# ------------------------------

def compute_pde_residual(xyt):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xyt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xyt)
            u = net_u(xyt)
            k = net_k(u)
        grads = tape1.gradient(u, xyt)             # [du/dx, du/dy, du/dt]
        du_dx, du_dy, du_dt = grads[:,0:1], grads[:,1:2], grads[:,2:3]
        flux_x = k * du_dx
        flux_y = k * du_dy
    flux_xx = tape2.gradient(flux_x, xyt)[:,0:1]
    flux_yy = tape2.gradient(flux_y, xyt)[:,1:2]
    return du_dt - (flux_xx + flux_yy)


# ------------------------------
# 8) Total loss
# ------------------------------

def compute_loss(xyt_domain, xyt_bc, u_bc, xyt_ic, u_ic, xyt_obs, u_obs, k_obs):
    # PDE residual loss
    res = compute_pde_residual(xyt_domain)
    loss_pde = tf.reduce_mean(tf.square(res))

    # IC loss
    u_pred_ic = net_u(xyt_ic)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic))

    # BC loss
    u_pred_bc = net_u(xyt_bc)
    loss_bc = tf.reduce_mean(tf.square(u_pred_bc - u_bc))

    # Supervised u at t=
    u_pred_obs = net_u(xyt_obs)
    loss_u  = tf.reduce_mean(tf.square(u_pred_obs - u_obs))

    # Supervised k(u) at t=
    k_pred_obs = net_k(u_pred_obs)
    loss_k  = tf.reduce_mean(tf.square(k_pred_obs - k_obs))

    
    total_loss = loss_pde + loss_ic + loss_bc + loss_u + 100 * loss_k

    return total_loss, loss_u, loss_k


# ------------------------------
# 9) Training loop (Adam)
# ------------------------------

batch_size_obs = 100
num_obs = XYT_obs_np.shape[0]
num_batches = int(np.ceil(num_obs / batch_size_obs))


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
epochs = 10_000

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
                xyt_bc     = xyt_bc, u_bc = u_bc,
                xyt_ic     = xyt_ic, u_ic = u_ic,
                xyt_obs    = XYT_obs_mb, u_obs = u_obs_mb, k_obs = k_obs_mb,
            )
        grads = tape.gradient(total_loss, net_u.trainable_variables + net_k.trainable_variables)
        optimizer.apply_gradients(zip(grads, net_u.trainable_variables + net_k.trainable_variables))

    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} — Total Loss: {total_loss.numpy():.4e} | Loss_u: {loss_u.numpy():.4e} | Loss_k: {loss_k.numpy():.4e}")

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** Training completed in {elapsed:.2f} seconds. ***")

# 1) Build a 100×100 grid at t = 
nx = ny = 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

times_to_save = [0.2, 0.4, 0.6, 0.8, 1.0]

os.makedirs("fields_over_time", exist_ok=True)

for t in times_to_save:
    T = t * np.ones_like(X)

    XYT_grid = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
    XYT_tensor = tf.convert_to_tensor(XYT_grid)

    # 2) Predict u and k
    u_tensor = net_u(XYT_tensor, training=False)
    k_tensor = net_k(u_tensor, training=False)

    # 3) Convert to NumPy and reshape
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


df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv("training_loss_log.csv", index=False)

# 5) Plot u(x,y,t)
plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, u_field, levels=100)
plt.colorbar(cf, label="u(x,y,t=1)")
plt.title("Predicted Temperature at t = ")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()

# 6) Plot k vs u scatter at t = 
plt.figure(figsize=(5,4))
plt.scatter(u_field, k_field, s=10, alpha=0.6)
plt.xlabel("u"); plt.ylabel("k(u)")
plt.title("Learned Conductivity vs Temperature (t = )")
plt.grid(True)
plt.tight_layout()

plt.show()
