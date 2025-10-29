#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import glob
import os

# ============================================
# Analytical Reference Solution
# ============================================
def U_exact(x):
    x_ = x[:, 0:1]
    t_ = x[:, 1:]
    u_exact = np.zeros_like(x_)
    for k in range(1, 6):
        u_exact += 1/10 * (
            np.cos((x_ - t_ + 0.5) * np.pi * k)
            + np.cos((x_ + t_ + 0.5) * np.pi * k)
        )
    return u_exact

# ============================================
# Load the Latest Saved Field
# ============================================
csv_files = sorted(glob.glob("field_epoch_*.csv"))
if not csv_files:
    raise FileNotFoundError("No field_epoch_*.csv files found in the current directory.")
latest_file = csv_files[-1]
print(f"Loading {latest_file} ...")
df = pd.read_csv(latest_file)

x = df["x"].values
t = df["t"].values
U_pred = df["u_pred"].values

# ============================================
# Create Structured Grid
# ============================================
x_unique = np.sort(df["x"].unique())
t_unique = np.sort(df["t"].unique())
X, T = np.meshgrid(x_unique, t_unique)
XT = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])

U_pred_grid = U_pred.reshape(len(t_unique), len(x_unique))
U_exact_grid = U_exact(XT).reshape(len(t_unique), len(x_unique))
U_error = U_pred_grid - U_exact_grid

# ============================================
# Select Time Slices
# ============================================
nslices = 5
time_indices = np.linspace(0, len(t_unique) - 1, nslices, dtype=int)
slice_times = t_unique[time_indices]

# ============================================
# Plot Layout
# ============================================
fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1.5, 1.5], wspace=0.25)

# --- Left: Predicted Field ---
ax_pred = fig.add_subplot(gs[0, 0])
im_pred = ax_pred.imshow(
    U_pred_grid,
    origin="lower",
    extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
    aspect="auto",
    cmap="RdBu_r"
)
ax_pred.set_xlabel("x (space)")
ax_pred.set_ylabel("t (time)")
ax_pred.set_title("PINN Predicted Field $U_{pred}(x,t)$")
plt.colorbar(im_pred, ax=ax_pred, label="U value")

# --- Middle: Analytical Solution ---
ax_exact = fig.add_subplot(gs[0, 1])
im_exact = ax_exact.imshow(
    U_exact_grid,
    origin="lower",
    extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
    aspect="auto",
    cmap="RdBu_r"
)
ax_exact.set_xlabel("x (space)")
ax_exact.set_ylabel("t (time)")
ax_exact.set_title("Analytical Solution $U_{exact}(x,t)$")
plt.colorbar(im_exact, ax=ax_exact, label="U value")

# --- Right: Error Map ---
ax_error = fig.add_subplot(gs[0, 2])
im_err = ax_error.imshow(
    U_error,
    origin="lower",
    extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
    aspect="auto",
    cmap="coolwarm"
)
ax_error.set_xlabel("x (space)")
ax_error.set_ylabel("t (time)")
ax_error.set_title("Error Field $U_{pred} - U_{exact}$")
plt.colorbar(im_err, ax=ax_error, label="Error")

plt.tight_layout()
plt.show()

# ============================================
# Also Plot Slices at Selected Times
# ============================================
plt.figure(figsize=(8, 6))
for i, idx in enumerate(time_indices):
    plt.plot(
        x_unique,
        U_exact_grid[idx, :],
        "k--",
        lw=1.5,
        label=f"Exact t={slice_times[i]:.2f}" if i == 0 else None
    )
    plt.plot(
        x_unique,
        U_pred_grid[idx, :],
        lw=2,
        label=f"PINN t={slice_times[i]:.2f}" if i == 0 else None
    )
plt.xlabel("x")
plt.ylabel("U(x, t)")
plt.title("Spatial Slices of PINN vs Exact Solution")
plt.legend()
plt.tight_layout()
plt.show()
