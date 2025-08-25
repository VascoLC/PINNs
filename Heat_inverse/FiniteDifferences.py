import numpy as np
import pandas as pd
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

Lx = 1
Nx = 1000
dx=Lx/Nx

x = np.linspace(0,1,Nx)

dt = 0.001
Nt = 1000

U0 = np.exp(-50*(x-0.5)**2) - np.exp(-50*(0.5)**2)
U = U0.copy()

def k(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

def build_laplacian(N, d):
    main = -2*np.ones(N)
    off  =  np.ones(N-1)
    return diags([main, off, off], [0,-1,1]) / d**2

Lx_op = build_laplacian(Nx, dx)
I = identity(Nx)

records = []
U_time = [] 
times_to_save = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
save_steps = {int(t / dt) for t in times_to_save} 
rng = np.random.default_rng(42)

for n in range(Nt):
    t_current = (n+1)*dt

    k_u = k(U)
    K_diag = diags(k_u)

    A = I - 0.5*dt*(K_diag @ Lx_op)
    B = (I + 0.5*dt*(K_diag @ Lx_op)) @ U

    U = spsolve(A, B)

    # Enforce Dirichlet BCs
    U[0] = 0
    U[-1] = 0

    step = n + 1
    t_current = step * dt

    # --- Save this time step? ---
    if step in save_steps:
        U_flat = U.copy()
        t_flat = np.full_like(x, t_current)

        idx = rng.choice(len(U_flat), size=50, replace=False)  # smaller sample in 1D

        df_step = pd.DataFrame({
            "x": x[idx],
            "t": t_flat[idx],
            "u": U_flat[idx],
            "k": k(U_flat[idx]),
        })
        records.append(df_step)
    U_time.append(U.copy())

U_time = np.array(U_time)  

# --- After loop, concatenate and write one big CSV ---
df_all = pd.concat(records, ignore_index=True)
df_all.to_csv("FiniteDifferences_data.csv", index=False)

# --- Plot: u(x) at final time ---
plt.figure(figsize=(6,4))
plt.plot(U, k(U), label=f't={Nt*dt:.1f}')
plt.xlabel("u(x,t)")
plt.ylabel("k(u)")
plt.title("K vs U")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.imshow(U_time, aspect='auto', extent=[x[0], x[-1],0, Nt*dt],origin='lower',cmap='inferno', vmin = 0.0, vmax = 1.0)
plt.colorbar(label='u(x, t)')

plt.show()