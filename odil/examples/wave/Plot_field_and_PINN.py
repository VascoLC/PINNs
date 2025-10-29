#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

def load_pickle(path):
    """Load ODIL pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_pinn_csv(path):
    """Load PINN predicted field from CSV."""
    df = pd.read_csv(path)
    x = df["x"].values
    t = df["t"].values
    U_pred = df["U_pred"].values
    # Identify unique grid values
    x_unique = np.sort(np.unique(x))
    t_unique = np.sort(np.unique(t))
    X, T = np.meshgrid(x_unique, t_unique)
    U_grid = U_pred.reshape(len(t_unique), len(x_unique))
    return X, T, U_grid

def plot_comparison(data, Xpinn, Tpinn, Upinn, save=False, outdir=".", prefix="compare"):
    """Plot Reference, ODIL, and PINN u(t,x) fields side-by-side."""
    # Extract ODIL arrays
    state_u = np.array(data["state_u"])   # ODIL predicted field
    ref_u   = np.array(data["ref_u"])     # Reference field
    lower   = np.array(data["lower"])
    upper   = np.array(data["upper"])
    Nt, Nx  = data["cshape"]

    # ODIL coordinate grid
    t_odil = np.linspace(lower[0], upper[0], Nt)
    x_odil = np.linspace(lower[1], upper[1], Nx)
    extent = [x_odil.min(), x_odil.max(), t_odil.min(), t_odil.max()]

    # Shared color scale
    vmax = np.max(np.abs(ref_u))
    vmin = -vmax

    # === Side-by-side contour plots ===
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    im0 = axs[0].imshow(ref_u, origin="lower", extent=extent, aspect="auto",
                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axs[0].set_title("Reference $u(t,x)$")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("t")

    im1 = axs[1].imshow(state_u, origin="lower", extent=extent, aspect="auto",
                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axs[1].set_title("ODIL Predicted $u(t,x)$")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("t")

    im2 = axs[2].imshow(Upinn, origin="lower",
                        extent=[Xpinn.min(), Xpinn.max(), Tpinn.min(), Tpinn.max()],
                        aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axs[2].set_title("PINN Predicted $u(t,x)$")
    axs[2].set_xlabel("x"); axs[2].set_ylabel("t")

    cbar = fig.colorbar(im0, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("$u(x,t)$", rotation=0, labelpad=10)

    vmin, vmax = -1.0, 1.0
    cbar.ax.text(1.2, 0.0, f"{vmin:.2f}", va="bottom", ha="left", transform=cbar.ax.transAxes)
    cbar.ax.text(1.2, 1.0, f"{vmax:.2f}", va="top", ha="left", transform=cbar.ax.transAxes)


    plt.tight_layout(rect=[0, 0, 0.88, 1])  


    if save:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{prefix}_fields.png")
        plt.savefig(outfile, dpi=300)
        print(f"Saved combined field plot to {outfile}")
    plt.show()

    # === Time-slice comparison ===
    # Select representative time slices (e.g., t = 0.0, 0.25, 0.5, 0.75, 1.0)
    t_slices = np.linspace(0, 1, 5)
    fig, axs = plt.subplots(1, len(t_slices), figsize=(18, 4), sharey=True)
    for i, ts in enumerate(t_slices):
        # Interpolate PINN field to this t
        idx_t_pinn = np.argmin(np.abs(Tpinn[:, 0] - ts))
        idx_t_odil = np.argmin(np.abs(t_odil - ts))
        u_ref = ref_u[idx_t_odil, :]
        u_odil = state_u[idx_t_odil, :]
        u_pinn = Upinn[idx_t_pinn, :]

        axs[i].plot(x_odil, u_ref, 'k-', label="Ref")
        axs[i].plot(x_odil, u_odil, 'r--', label="ODIL")
        axs[i].plot(Xpinn[0, :], u_pinn, 'b-.', label="PINN")
        axs[i].set_title(f"t = {ts:.2f}")
        axs[i].set_xlabel("x")
        if i == 0:
            axs[i].set_ylabel("u(x,t)")
        axs[i].grid(True)
    axs[0].legend()
    plt.suptitle("Reference vs ODIL vs PINN across time slices", y=1.05)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, wspace=0.25)
    if save:
        outfile = os.path.join(outdir, f"{prefix}_timeslices.png")
        plt.savefig(outfile, dpi=300)
        print(f"Saved time-slice comparison to {outfile}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ODIL, PINN, and Reference fields.")
    parser.add_argument("--odil_pickle", default="out_wave_lbfgsb/data_00008.pickle", help="ODIL pickle file path")
    parser.add_argument("--pinn_csv", default="C:\PINNs_Git\PINNs\Wave\Wave_Results_Fourier_40k/predicted_field.csv", help="PINN CSV file path")
    parser.add_argument("--save", action="store_true", help="Save plots")
    parser.add_argument("--outdir", default=".", help="Output directory for saved plots")
    args = parser.parse_args()

    data = load_pickle(args.odil_pickle)
    Xpinn, Tpinn, Upinn = load_pinn_csv(args.pinn_csv)

    plot_comparison(data, Xpinn, Tpinn, Upinn, save=args.save, outdir=args.outdir)
