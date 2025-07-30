import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # <-- for CSV

Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

steps_to_save = [0.2, 0.4, 0.6, 0.8, 1.0]

def apply_bc(u):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    return u

def compute_div(u, k_u, dx, dy):
    k_ip = 0.5 * (k_u[1:, :] + k_u[:-1, :])
    fx = k_ip * (u[1:, :] - u[:-1, :]) / dx

    k_jp = 0.5 * (k_u[:, 1:] + k_u[:, :-1])
    fy = k_jp * (u[:, 1:] - u[:, :-1]) / dy

    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (
        (fx[1:,   1:-1] - fx[:-1, 1:-1]) / dx +
        (fy[1:-1, 1:]   - fy[1:-1, :-1]) / dy
    )
    return div

def solve_cn(u0, k_func, Lx, Ly, Nx, Ny, dt, Nt, save_times=None, tol=1e-6, max_iter=50):
    dx = Lx / Nx
    dy = Ly / Ny
    u = u0.copy()
    apply_bc(u)

    snapshots = [u.copy()]
    times = [0.0]

    if save_times is None:
        save_steps = set()
    else:
        save_steps = {int(round(t / dt)) for t in save_times}

    for n in range(Nt):
        u_prev = u.copy()
        apply_bc(u_prev)
        k_prev = k_func(u_prev)
        L_prev = compute_div(u_prev, k_prev, dx, dy)

        u_new = u_prev.copy()
        for _ in range(max_iter):
            apply_bc(u_new)
            k_new = k_func(u_new)
            L_new = compute_div(u_new, k_new, dx, dy)

            u_next = u_prev + 0.5 * dt * (L_prev + L_new)
            if np.max(np.abs(u_next - u_new)) < tol:
                u_new = u_next
                break
            u_new = u_next

        u = apply_bc(u_new)

        if (n+1) in save_steps:
            snapshots.append(u.copy())
            times.append((n+1) * dt)

    return snapshots, times

# 4. Initial condition
g0 = np.exp(-50*(0.5)**2)
u0 = np.exp(-50*(X - 0.5)**2) - g0

# 5. Conductivity law
k_func = lambda u: 0.02 * np.exp(-20*(u-0.5)**2)

# 6. Time-stepping parameters
dt = 0.25 * min((Lx/Nx)**2, (Ly/Ny)**2)
T = 1
Nt = int(T / dt)

# 7. Run solver
snaps, ts = solve_cn(u0, k_func, Lx, Ly, Nx, Ny, dt, Nt, save_times=steps_to_save)

# snaps[i] is the field at time ts[i]
for t, u_snap in zip(ts, snaps):
    print(f"Saved snapshot at t = {t:.2f}")

# Plot a 2×3 grid
fig, axes = plt.subplots(2, 3, figsize=(12,8))
for ax, u_snap, t in zip(axes.flat, snaps, ts):
    im = ax.imshow(u_snap.T, origin="lower",
                   extent=[0,Lx,0,Ly], vmin=0)
    ax.set_title(f"t = {t:.1f}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('u')

plt.tight_layout()
plt.show()

### ---- CSV OUTPUT SECTION ---- ###
frames = []
rng = np.random.default_rng(42)
for u_snap, t in zip(snaps, ts):
    k_snap = k_func(u_snap)
    x_flat = X.ravel()
    y_flat = Y.ravel()
    u_flat = u_snap.ravel()
    k_flat = k_snap.ravel()
    t_flat = np.full_like(u_flat, fill_value=t, dtype=float)
    Ntotal = x_flat.size
    idx = rng.choice(Ntotal, size=100, replace=False)

    df = pd.DataFrame({
        'x': x_flat[idx],
        'y': y_flat[idx],
        't': np.full(100, t),
        'u': u_flat[idx],
        'k': k_flat[idx]
    })
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)
all_data.to_csv('data.csv', index=False)
print(f"Written {len(all_data)} rows to snapshots.csv")
