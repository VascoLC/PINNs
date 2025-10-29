import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import math

#print("Number of CPU threads:", torch.get_num_threads())
# ======================================
# Helper Function
# ======================================
def get_ref_k(u, mod=np):
    """Reference conductivity: Gaussian."""
    return 0.02 * np.exp(-((u - 0.5) ** 2) * 20)

# ======================================
# Neural Networks
# ======================================
class NetU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(2, 32), nn.Tanh()])
        for _ in range(3):  # 4 hidden layers total
            self.hidden.append(nn.Linear(32, 32))
            self.hidden.append(nn.Tanh())
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

class NetK(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(1, 5), nn.Tanh(), nn.Linear(5, 5), nn.Tanh()])
        self.out = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u):
        u_norm = 2.0 * (u - 0.5)
        x = u_norm
        for layer in self.hidden:
            x = layer(x)
        z = self.sigmoid(self.out(x))
        return 0.1 * z

# ======================================
# Data Loading
# ======================================
df = pd.read_csv(r"C:\PINNs_Git\PINNs\odil\examples\heat/1D/out_heat_inverse_1D_Imposed_solution\imposed_data_with_u.csv")

XT_obs_np = df[["x", "t"]].values.astype(np.float32)
u_obs_np  = df[["u"]].values.reshape(-1, 1).astype(np.float32)
k_obs_np  = get_ref_k(u_obs_np).astype(np.float32)

XT_obs = torch.tensor(XT_obs_np)
u_obs  = torch.tensor(u_obs_np)
k_obs  = torch.tensor(k_obs_np)

# Initial condition
n_ic = 400
x_ic = torch.rand(n_ic, 1)
t_ic = torch.zeros_like(x_ic)
xt_ic = torch.cat([x_ic, t_ic], dim=1)
u_ic = torch.exp(-50 * (x_ic - 0.5) ** 2) - math.exp(-50 * 0.5**2)

# Boundary condition
n_bc = 400
t_bc = torch.rand(n_bc, 1)
edges = torch.randint(0, 2, (n_bc,))  # 0=left, 1=right
x_bc = torch.rand(n_bc, 1)
x_bc[edges == 0] = 0.0
x_bc[edges == 1] = 1.0
xt_bc = torch.cat([x_bc, t_bc], dim=1)
u_bc = torch.zeros_like(x_bc)

# Domain samples for PDE residual
n_dom = 4096
x_dom = torch.rand(n_dom, 1)
t_dom = torch.rand(n_dom, 1)
xt_dom = torch.cat([x_dom, t_dom], dim=1)

# ======================================
# Instantiate Models
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net_u = NetU().to(device)
net_k = NetK().to(device)

# ======================================
# PDE Residual
# ======================================
def compute_pde_residual(xt):
    xt = xt.requires_grad_(True)
    u = net_u(xt)
    k = net_k(u)

    du = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_dt = du[:, 0:1], du[:, 1:2]
    flux_x = k * du_dx

    dflux = torch.autograd.grad(flux_x, xt, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0]
    flux_xx = dflux[:, 0:1]
    return du_dt - flux_xx

# ======================================
# Total Loss
# ======================================
def compute_loss():
    # PDE residual loss
    res = compute_pde_residual(xt_dom.to(device))
    loss_pde = torch.mean(res**2)

    # IC loss
    u_pred_ic = net_u(xt_ic.to(device))
    loss_ic = torch.mean((u_pred_ic - u_ic.to(device))**2)

    # BC loss
    u_pred_bc = net_u(xt_bc.to(device))
    loss_bc = torch.mean((u_pred_bc - u_bc.to(device))**2)

    # Supervised u
    u_pred_obs = net_u(XT_obs.to(device))
    loss_u = torch.mean((u_pred_obs - u_obs.to(device))**2)

    # Supervised k
    k_pred_obs = net_k(u_pred_obs)
    loss_k = torch.mean((k_pred_obs - k_obs.to(device))**2)

    total_loss = loss_pde + loss_ic + loss_bc + loss_u + loss_k
    return total_loss, loss_u, loss_k

# ======================================
# Plotting Function
# ======================================
def plot_and_save(epoch, folder_name):
    Nx, Nt = 200, 200
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t)
    XT_grid = np.hstack([X.flatten()[:, None], T.flatten()[:, None]]).astype(np.float32)

    with torch.no_grad():
        XT_tensor = torch.tensor(XT_grid, device=device)
        u_pred = net_u(XT_tensor).cpu().numpy().reshape(Nt, Nx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cf = ax1.contourf(X, T, u_pred, levels=100, cmap="viridis")
    fig.colorbar(cf, ax=ax1, label="u(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title(f"u(x,t) at Epoch {epoch}")

    # k(u)
    u_range = np.linspace(0, 1, 400).reshape(-1, 1).astype(np.float32)
    k_true = get_ref_k(u_range)
    with torch.no_grad():
        k_learned = net_k(torch.tensor(u_range, device=device)).cpu().numpy()
    ax2.plot(u_range, k_true, "r--", label="True k(u)")
    ax2.plot(u_range, k_learned, "b-", label="Learned k(u)")
    ax2.legend()
    ax2.set_xlabel("u")
    ax2.set_ylabel("k(u)")
    ax2.set_title("Conductivity k(u)")
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f"results_epoch_{str(epoch).zfill(6)}.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")

# ======================================
# Training
# ======================================
# Phase 1: Adam
loss_history = []
start_time = time.perf_counter()
optimizer = torch.optim.Adam(list(net_u.parameters()) + list(net_k.parameters()), lr=1e-3)
epoch = [0]
for epoch in range(200_000):
    optimizer.zero_grad()
    total_loss, loss_u, loss_k = compute_loss()
    total_loss.backward()
    optimizer.step()

    loss_history.append([epoch, total_loss.item(), loss_u.item(), loss_k.item()])

    if epoch % 100 == 0:
        print(f"Epoch {epoch:05d} | Loss = {total_loss.item():.4e} | Loss_u = {loss_u.item():.4e} | Loss_k = {loss_k.item():.4e}")


# Phase 2: L-BFGS
optimizer = torch.optim.LBFGS(
    list(net_u.parameters()) + list(net_k.parameters()),
    lr=1.0,
    max_iter=200_000,
    tolerance_grad=1e-12,
    tolerance_change=1e-12,
    history_size=100
)

iteration = [0]

def closure():
    optimizer.zero_grad()
    total_loss, loss_u, loss_k = compute_loss()
    total_loss.backward()
    loss_history.append([epoch + iteration[0], total_loss.item(), loss_u.item(), loss_k.item()])
    if iteration[0] % 100 == 0:
        print(f"L-BFGS iter {iteration[0]:05d} | Loss = {total_loss.item():.4e}| Loss_u = {loss_u.item():.4e} | Loss_k = {loss_k.item():.4e}")
    iteration[0] += 1
    return total_loss


optimizer.step(closure)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"\n*** Training completed in {elapsed:.2f} seconds. ***")

# ======================================
# Final Output
# ======================================
Nx, Nt = 500, 500
x_values = np.linspace(0, 1, Nx)
t_values = np.linspace(0, 1, Nt)
Xg, Tg = np.meshgrid(x_values, t_values)
XT_grid = np.hstack([Xg.flatten()[:, None], Tg.flatten()[:, None]]).astype(np.float32)
XT_tensor = torch.tensor(XT_grid, device=device)

with torch.no_grad():
    u_pred = net_u(XT_tensor).cpu().numpy().reshape(Nt, Nx)
    k_pred = net_k(net_u(XT_tensor)).cpu().numpy().reshape(Nt, Nx)

df_final = pd.DataFrame({
    "x": Xg.flatten(),
    "t": Tg.flatten(),
    "u": u_pred.flatten(),
    "k": k_pred.flatten(),
})
df_final.to_csv("Heat_Inverse_1D_solution_PyTorch.csv", index=False)

df_loss = pd.DataFrame(loss_history, columns=["epoch", "total_loss", "loss_u", "loss_k"])
df_loss.to_csv("training_loss_log_PyTorch.csv", index=False)

plt.figure(figsize=(7, 15))
cf = plt.contourf(Xg, Tg, u_pred, levels=100, cmap="viridis")
plt.colorbar(cf, label="u(x,t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN solution $u(x,t)$")
plt.tight_layout()
plt.show()
