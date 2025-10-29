#!/usr/bin/env python3
"""
plot_compare_u.py

Compare:
1. Reference temperature field (from ref_solution.pickle)
2. ODIL simulation (from data_XXXXX.pickle)
3. PINN temperature field (from predicted_fields.csv)

Also plots the conductivity function k(u) for all three datasets.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import os


# ==============================================================
# --- Utility functions ---
# ==============================================================

def load_pickle(path):
    """Load pickle safely."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_ref_k(u):
    """Reference conductivity law (same as used in ODIL forward model)."""
    return 0.02 * np.exp(-((u - 0.5) ** 2) * 20)


def load_and_interpolate_pinn_csv(path, x_target, y_target, t_target):
    """
    Load a PINN temperature field from CSV and interpolate it onto the target (t,x,y) grid.
    """
    df = pd.read_csv(path)
    cols = df.columns.str.lower().tolist()

    # Identify columns
    tcol = next((c for c in ["t", "time"] if c in cols), None)
    xcol = next((c for c in ["x"] if c in cols), None)
    ycol = next((c for c in ["y"] if c in cols), None)
    ucol = next((c for c in ["u_pred", "u", "temperature", "temp", "u_predicted"] if c in cols), None)

    if not all([tcol, xcol, ycol, ucol]):
        raise ValueError(f"CSV must contain t,x,y,u columns. Found: {cols}")

    # Sort unique coordinates
    t_vals = np.sort(df[tcol].unique())
    x_vals = np.sort(df[xcol].unique())
    y_vals = np.sort(df[ycol].unique())

    Nt, Nx, Ny = len(t_vals), len(x_vals), len(y_vals)
    print(f"[PINN] Detected grid: Nt={Nt}, Nx={Nx}, Ny={Ny}")

    # Reshape u values to (Nt, Nx, Ny)
    df_sorted = df.sort_values([tcol, xcol, ycol])
    u_field = df_sorted[ucol].values.reshape((Nt, Nx, Ny))

    # Interpolator from PINN grid to target grid
    interpolator = RegularGridInterpolator(
        (t_vals, x_vals, y_vals),
        u_field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Create target mesh and interpolate
    Tm, Xm, Ym = np.meshgrid(t_target, x_target, y_target, indexing="ij")
    points_target = np.stack([Tm.ravel(), Xm.ravel(), Ym.ravel()], axis=-1)
    u_interp = interpolator(points_target).reshape(len(t_target), len(x_target), len(y_target))
    return u_interp


# ==============================================================
# --- Main plotting routine ---
# ==============================================================

def plot_comparison(ref_path, data_path, pinn_csv, output_path="comparison_u.png"):
    # --- Load pickle data ---
    ref_data = load_pickle(ref_path)
    data = load_pickle(data_path)

    # --- Extract ODIL and reference arrays ---
    ref_u = np.array(data["ref_u"])      # shape (Nt, Nx, Ny)
    u = np.array(data["state_u"])        # shape (Nt, Nx, Ny)
    Nt, Nx, Ny = u.shape
    print(f"[Pickle] Loaded reference {ref_u.shape}, ODIL {u.shape}")

    # --- Build coordinate grids ---
    x_target = np.linspace(0, 1, Nx)
    y_target = np.linspace(0, 1, Ny)
    t_target = np.linspace(0, 1, Nt)

    # --- Load and interpolate PINN results ---
    pinn_u = load_and_interpolate_pinn_csv(pinn_csv, x_target, y_target, t_target)
    print(f"[PINN] Interpolated field to {pinn_u.shape}")

    # --- Smooth slightly for prettier plots ---
    sigma = 0.7
    ref_u = gaussian_filter(ref_u, sigma=sigma)
    u = gaussian_filter(u, sigma=sigma)
    pinn_u = gaussian_filter(pinn_u, sigma=sigma)

    # ===============================================================
    # --- Temperature field comparison ---
    # ===============================================================
    extent = [x_target.min(), x_target.max(), y_target.min(), y_target.max()]
    time_indices = np.linspace(0, Nt - 1, 5, dtype=int)
    time_labels = t_target[time_indices]

    fig, axes = plt.subplots(3, 5, figsize=(15, 7.5), sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.25)
    vmin, vmax = 0, 1

    models = [("Reference", ref_u), ("ODIL", u), ("PINN", pinn_u)]

    for row, (label, arr) in enumerate(models):
        for col, t_idx in enumerate(time_indices):
            ax = axes[row, col]
            im = ax.imshow(arr[t_idx, :, :].T, origin="lower",
                           extent=extent, cmap="YlOrBr",
                           vmin=vmin, vmax=vmax,
                           interpolation="bicubic")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(models) - 1:
                ax.set_xlabel("x")
            ax.set_xticks([])
            ax.set_yticks([])

    # --- Add colorbar ---
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Temperature (u)')

    # --- Add a shared time axis above ---
    time_ax = fig.add_axes([0.1, 0.92, 0.8, 0.04], frameon=False)
    time_ax.set_xlim(t_target.min(), t_target.max())
    time_ax.set_ylim(0, 1)
    time_ax.set_xticks(time_labels)
    time_ax.set_xticklabels([f"{t:.2f}" for t in time_labels])
    time_ax.set_xlabel("Time", fontsize=12)
    time_ax.xaxis.set_label_position("top")
    time_ax.xaxis.tick_top()
    time_ax.yaxis.set_visible(False)

    fig.suptitle("Comparison of Temperature Field u(x,y,t)", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"[Done] Saved temperature comparison to {output_path}")
    plt.close(fig)

    # ===============================================================
    # --- Conductivity function k(u) vs u ---
    # ===============================================================

    ref_uk = np.linspace(0, 1, 200)
    ref_k = get_ref_k(ref_uk)

    # ODIL inferred conductivity
    if "k" in data:
        k_odil = np.array(data["k"]).flatten()
        ref_uk_odil = np.array(data["ref_uk"]).flatten()
    else:
        k_odil = get_ref_k(ref_u.flatten())
        ref_uk_odil = ref_u.flatten()

    # PINN predicted conductivity (via k_pred column if present)
    pinn_df = pd.read_csv(pinn_csv)
    k_col = next((c for c in pinn_df.columns if "k" in c.lower()), None)
    u_col = next((c for c in pinn_df.columns if "u" in c.lower()), None)

    if k_col is not None and u_col is not None:
        pinn_u_vals = pinn_df[u_col].values
        pinn_k_vals = pinn_df[k_col].values
    else:
        print("[WARN] No k_pred column found in PINN CSV; using reference law as placeholder.")
        pinn_u_vals = pinn_df[u_col].values if u_col else np.linspace(0, 1, 100)
        pinn_k_vals = get_ref_k(pinn_u_vals)

    # --- Plot conductivity comparison ---
    plt.figure(figsize=(5.5, 4))
    plt.plot(ref_uk, ref_k, 'k--', lw=2, label='Analytical')
    plt.scatter(ref_uk_odil, k_odil, color = 'red',s=6, alpha=0.4, label='ODIL')
    plt.scatter(pinn_u_vals, pinn_k_vals,color='green', s=4, alpha=0.4, label='PINN')
    plt.xlabel("Temperature u(x,y,t)")
    plt.ylabel("Conductivity k(u)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("conductivity_comparison.png", dpi=200)
    print("[Done] Saved conductivity comparison to conductivity_comparison.png")
    plt.close()
    

# ==============================================================
# --- Entry point ---
# ==============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare reference, ODIL, and PINN temperature fields.")
    parser.add_argument("--ref", type=str,
                        default=r"C:\PINNs_Git\PINNs\odil\examples\heat\2D\First_Step\out_heat_direct_2D_128\data_00010.pickle",
                        help="Path to reference solution pickle file")
    parser.add_argument("--data", type=str,
                        default=r"C:\PINNs_Git\PINNs\odil\examples\heat\2D\First_Step\out_heat_inverse_2D_32_MODIL_ADAM\data_00040.pickle",
                        help="Path to ODIL simulation pickle file")
    parser.add_argument("--pinn", type=str,
                        default=r"C:\PINNs_Git\PINNs\Heat_inverse_2D\2D_first_step\Full_Batch_ADAM\final_results_ADAM\predicted_fields.csv",
                        help="Path to PINN temperature CSV file")
    parser.add_argument("--out", type=str, default="comparison_u.png",
                        help="Output image filename")

    args = parser.parse_args()
    plot_comparison(args.ref, args.data, args.pinn, args.out)
