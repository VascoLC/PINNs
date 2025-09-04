import numpy as np
import pandas as pd
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- Domain and grid ---
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# --- Time parameters ---
dt = 0.001
Nt = 1000

# --- Initial condition ---
U0 = 16*X*(1-X)*Y*(1-Y)
U = U0.flatten()

# --- Nonlinear conductivity ---
def k(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

# --- Build operators ---
def build_laplacian(N, d):
    main = -2*np.ones(N)
    off  =  np.ones(N-1)
    return diags([main, off, off], [0,-1,1]) / d**2

Lx_op = build_laplacian(Nx, dx)
Ly_op = build_laplacian(Ny, dy)
Ix, Iy = identity(Nx), identity(Ny)
Laplacian = kron(Iy, Lx_op) + kron(Ly_op, Ix)
I = identity(Nx*Ny)

# --- Prepare storage for every time step ---
records = []

times_to_save = [0.2, 0.4, 0.6, 0.8, 1.0]
save_steps = {int(t / dt) for t in times_to_save}  # {200, 400, 600, 800, 1000}
rng = np.random.default_rng(42)

# --- Time stepping loop ---
for n in range(Nt):
    t_current = (n+1)*dt

    k_u = k(U)
    K_diag = diags(k_u)

    A = I - 0.5*dt*(K_diag @ Laplacian)
    B = (I + 0.5*dt*(K_diag @ Laplacian)) @ U

    U = spsolve(A, B)

    # Enforce Dirichlet BCs
    U_mat = U.reshape(Ny, Nx)
    U_mat[0, :] = 0
    U_mat[-1, :] = 0
    U_mat[:, 0] = 0
    U_mat[:, -1] = 0
    U = U_mat.flatten()

    step = n + 1
    t_current = step * dt

    # --- Save this time step? ---
    if step in save_steps:
        U_flat = U.copy()
        # flatten coords
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        # create a t‐column of shape (Nx*Ny,)
        t_flat = np.full_like(X_flat, t_current)

        idx = rng.choice(len(U_flat), size=500, replace=False)
        
        df_step = pd.DataFrame({
            "x": X_flat[idx],
            "y": Y_flat[idx],
            "t": t_flat[idx],
            "u": U_flat[idx],
            "k": k(U_flat[idx]),
        })
        records.append(df_step)

# --- After loop, concatenate and write one big CSV ---
df_all = pd.concat(records, ignore_index=True)
df_all.to_csv("FD_train_data_2500_points.csv", index=False)


# --- After the time‐stepping loop, U contains u at t = Nt*dt ---
U_final = U.reshape(Ny, Nx)

# --- Plot as a filled contour ---
plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, U_final, levels=100)
plt.colorbar(cf, label="Temperature u(x,y,t=1.0)", cmap = "inferno")
plt.title(f"Temperature Field at t = {Nt*dt:.1f}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()
plt.show()
