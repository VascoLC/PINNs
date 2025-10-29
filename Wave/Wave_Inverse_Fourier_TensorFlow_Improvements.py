import tensorflow as tf
import numpy as np
import pandas as pd

# ------------------------------
# Neural Networks Architecture
# ------------------------------
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, m_x=32, m_t=32, sigmas_x=(1), sigmas_t=(1.0, 2.0, 5.0, 10), seed=42, dtype=tf.float64):
        super().__init__(dtype=dtype)
        rng = np.random.RandomState(seed)
        # Build stacked projection matrices: shape [in_dim, total_features]
        Bx_parts, Bt_parts = [], []
        for s in sigmas_x:
            Bx_parts.append(rng.normal(loc=0.0, scale=s, size=(1, m_x)))
        for s in sigmas_t:
            Bt_parts.append(rng.normal(loc=0.0, scale=s, size=(1, m_t)))
        Bx = np.concatenate(Bx_parts, axis=1)  # [1, m_x * len(sigmas_x)]
        Bt = np.concatenate(Bt_parts, axis=1)  # [1, m_t * len(sigmas_t)]
        self.Bx = tf.constant(Bx, dtype=dtype)
        self.Bt = tf.constant(Bt, dtype=dtype)

    def call(self, x):
        # x: [..., 2] with columns [x, t]
        x = tf.cast(x, self.dtype)
        x_spatial = x[:, 0:1]                                   # [N,1]
        t_temporal = x[:, 1:2]                                  # [N,1]
        proj_x = tf.matmul(x_spatial, self.Bx)                  # [N, Mx]
        proj_t = tf.matmul(t_temporal, self.Bt)                 # [N, Mt]
        fx = tf.concat([tf.sin(2*np.pi*proj_x), tf.cos(2*np.pi*proj_x)], axis=1)
        ft = tf.concat([tf.sin(2*np.pi*proj_t), tf.cos(2*np.pi*proj_t)], axis=1)
        return tf.concat([x, fx, ft], axis=1)                   # include raw x,t too


tf.keras.backend.set_floatx("float64")

class FourierPINN(tf.keras.Model):
    def __init__(self):
        super().__init__(dtype=tf.float64)
        self.fourier = FourierLayer(m_x=32, m_t=32,
                                    sigmas_x=(0.5,1,2,5),
                                    sigmas_t=(0.5,1,2,5))
        self.hidden = [tf.keras.layers.Dense(128, activation="tanh", dtype=tf.float64) for _ in range(3)]
        self.out = tf.keras.layers.Dense(1, dtype=tf.float64)

    def call(self, x):
        x = self.fourier(x)
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

    
# ------------------------------
# Boundary and Initial conditions
# ------------------------------

def U_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:]
    u_exact = np.zeros_like(x_)
    for k in range(1,6):
        u_exact+=1/10*(np.cos((x_-t_+0.5)*np.pi*k)+np.cos((x_+t_+0.5)*np.pi*k))
    return u_exact

def Ut_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:1+1]
    ut = np.zeros_like(x_)
    for k in range(1,6):
        ut += 1/10 * np.pi * k * (
            np.sin((x_ - t_ + 0.5)*np.pi*k) -
            np.sin((x_ + t_ + 0.5)*np.pi*k)
        )
    return ut


# Initial condition
x_ic = np.linspace(-1, 1, 400)[:, None]
t_ic = np.zeros_like(x_ic)
init_xt = np.hstack([x_ic, t_ic])
init_u = U_exact(init_xt)
init_ut = Ut_exact(init_xt)
xt_ic = tf.convert_to_tensor(init_xt,dtype=tf.float64)
u_ic   = tf.convert_to_tensor(init_u,dtype=tf.float64)
ut_ic   = tf.convert_to_tensor(init_ut,dtype=tf.float64)

# Left and right boundaries
t_bc = np.linspace(0, 1, 400)[:, None]
x_bc_left  = -np.ones_like(t_bc)
x_bc_right =  np.ones_like(t_bc)
l_bc_np = np.hstack([x_bc_left, t_bc])
r_bc_np = np.hstack([x_bc_right, t_bc])
l_bc = tf.convert_to_tensor(l_bc_np,dtype=tf.float64)
r_bc = tf.convert_to_tensor(r_bc_np,dtype=tf.float64)
u_bc_left  = U_exact(l_bc_np)
u_bc_right = U_exact(r_bc_np)
u_bc_left_tf  = tf.convert_to_tensor(u_bc_left, dtype=tf.float64)
u_bc_right_tf = tf.convert_to_tensor(u_bc_right, dtype=tf.float64)

# Analytical collocation points
n_obs = 500
x_obs = np.random.uniform(-1, 1, (n_obs, 1))
t_obs = np.random.uniform(0, 1, (n_obs, 1))
xt_obs_np = np.hstack([x_obs, t_obs])
u_obs_np  = U_exact(xt_obs_np) 
xt_obs = tf.convert_to_tensor(xt_obs_np, dtype=tf.float64)
u_obs  = tf.convert_to_tensor(u_obs_np, dtype=tf.float64)


# Domain collocation points
n_dom = 4096
x_dom = np.random.uniform(-1, 1, (n_dom, 1))
t_dom = np.random.uniform(0, 1, (n_dom, 1))
xt_dom_np = np.hstack([x_dom, t_dom])
xt_dom = tf.convert_to_tensor(xt_dom_np,dtype=tf.float64)

# PDE residual
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
# Instanciate NN
# ------------------------------
net_u = FourierPINN()

# ------------------------------
# Optimisation Loss
# ------------------------------
W = dict(pde=1.0, ic1=10.0, ic2=10.0, lbc=10.0, rbc=10.0, data=1.0)

def compute_loss(xt_domain, l_bc, r_bc, xt_ic, u_ic, ut_ic, u_bc_l, u_bc_r, xt_obs, u_obs):
    loss_pde = tf.reduce_mean(tf.square(pde_residual(net_u, xt_domain)))
    loss_ic1 = tf.reduce_mean(tf.square(net_u(xt_ic)-u_ic))

    with tf.GradientTape() as tape1:
        tape1.watch(xt_ic)
        u_ic_pred = net_u(xt_ic)
    ut_ic_pred = tape1.gradient(u_ic_pred, xt_ic)[:, 1:2]
    loss_ic2 = tf.reduce_mean(tf.square(ut_ic_pred - ut_ic))

    loss_lbc = tf.reduce_mean(tf.square(net_u(l_bc) - u_bc_l))
    loss_rbc = tf.reduce_mean(tf.square(net_u(r_bc) - u_bc_r))
    loss_obs = tf.reduce_mean(tf.square(net_u(xt_obs) - u_obs))

    total = (W["pde"]*loss_pde + W["ic1"]*loss_ic1 + W["ic2"]*loss_ic2 +
             W["lbc"]*loss_lbc + W["rbc"]*loss_rbc + W["data"]*loss_obs)

    return total, {"pde": loss_pde, "ic1": loss_ic1, "ic2": loss_ic2, "lbc": loss_lbc, "rbc": loss_rbc, "data": loss_obs}


# ------------------------------
# Training Loop
# ------------------------------
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=5000, decay_rate=0.5, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
epochs = 125_000
loss_history = []

# Grid for visualization
nx, nt = 200, 200
x_lin = np.linspace(-1, 1, nx)
t_lin = np.linspace(0, 1, nt)
X, T = np.meshgrid(x_lin, t_lin)
XT_grid = np.hstack([X.flatten()[:, None], T.flatten()[:, None]]).astype(np.float32)
XT_tensor = tf.convert_to_tensor(XT_grid, dtype=tf.float32)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        total_loss, losses = compute_loss(
            xt_dom, l_bc, r_bc, xt_ic, u_ic, ut_ic,
            u_bc_left_tf, u_bc_right_tf,
            xt_obs, u_obs
        )
    grads = tape.gradient(total_loss, net_u.trainable_variables)
    optimizer.apply_gradients(zip(grads, net_u.trainable_variables))
    loss_history.append([epoch,float(total_loss),
        float(losses["pde"]),
        float(losses["ic1"]),
        float(losses["ic2"]),
        float(losses["lbc"]),
        float(losses["rbc"]),
        float(losses["data"]),
    ])

    if epoch % 500 == 0:
            print(f"Epoch {epoch:05d} | Total Loss: {total_loss.numpy():.3e} | PDE: {losses['pde'].numpy():.3e}")
            u_pred = net_u(XT_tensor).numpy().flatten()
            df_field = pd.DataFrame({"x": XT_grid[:, 0],"t": XT_grid[:, 1],"u_pred": u_pred})
            filename = f"field_epoch_{epoch:06d}.csv"
            df_field.to_csv(filename, index=False)
            columns = ["epoch", "total", "pde", "ic1", "ic2", "lbc", "rbc", "obs_u"]
            loss_df = pd.DataFrame(loss_history, columns=columns)
            loss_df.to_csv("loss_history.csv", index=False)
            print("Saved loss history to loss_history.csv")

