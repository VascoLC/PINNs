#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_pickle(path):
    """Load ODIL pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def plot_u(data, save=False, outdir=".", prefix="field"):
    """
    Plot ODIL reference and predicted fields u(t,x)
    from a data_xxxxx.pickle file.
    """

    # Extract arrays
    state_u = np.array(data["state_u"])   # predicted field
    ref_u   = np.array(data["ref_u"])     # reference field
    lower   = np.array(data["lower"])     # (t_min, x_min)
    upper   = np.array(data["upper"])     # (t_max, x_max)
    Nt, Nx  = data["cshape"]

    # Create physical coordinate grids
    t = np.linspace(lower[0], upper[0], Nt)
    x = np.linspace(lower[1], upper[1], Nx)
    extent = [x.min(), x.max(), t.min(), t.max()]

    # Shared color scale (symmetric)
    v = np.max(np.abs(ref_u))

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    im0 = axs[0].imshow(
        ref_u, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-v, vmax=v
    )
    axs[0].set_title("Reference $u(t,x)$")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("t")

    im1 = axs[1].imshow(
        state_u, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-v, vmax=v
    )
    axs[1].set_title("ODIL Predicted $u(t,x)$")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("t")

    fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.8, label="$u$ value")

    plt.tight_layout()
    if save:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{prefix}_u.png")
        plt.savefig(outfile, dpi=300)
        print(f"Saved plot to {outfile}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ODIL u field from a data_xxxxx.pickle file."
    )
    parser.add_argument("--pickle_file",default='C:\PINNs_Git\PINNs\odil\examples\wave\out_wave_lbfgsb_M/data_00008.pickle', help="Path to ODIL data_XXXXX.pickle file")
    parser.add_argument("--save", action="store_true", help="Save plot to file")
    parser.add_argument("--outdir", default=".", help="Output directory for saved figure")
    args = parser.parse_args()

    data = load_pickle(args.pickle_file)
    plot_u(data, save=args.save, outdir=args.outdir)
