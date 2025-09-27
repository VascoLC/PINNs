#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

import odil
from odil import Field
from odil.core import checkpoint_load

def main():
    parser = argparse.ArgumentParser(
        description="Load an ODIL ref_solution.pickle and plot u at t=0:0.2:1"
    )
    parser.add_argument(
        "--pickle", "-p",
        default="out_heat2D_direct_final_step/ref_solution.pickle",
        help="path to your ODIL checkpoint (ref_solution.pickle)"
    )
    parser.add_argument(
        "--Nx", type=int,
        default=64,
        help="number of x-points in your run"
    )
    parser.add_argument(
        "--Ny", type=int,
        default=64,
        help="number of y-points in your run"
    )
    parser.add_argument(
        "--Nt", type=int,
        default=64,
        help="number of t-points in your run"
    )
    args = parser.parse_args()

    # Rebuild the same Domain you used for direct solve
    domain = odil.Domain(
        cshape=(args.Nt, args.Nx, args.Ny),
        dimnames=("t", "x", "y"),
        lower=(0.0, 0.0, 0.0),
        upper=(1.0, 1.0, 1.0),
        multigrid=False,
        dtype=np.float64
    )

    # Load the saved state
    state = odil.State(fields={"u": Field()})
    checkpoint_load(domain, state, args.pickle)

    # Extract into a numpy array
    u_ref = np.array(domain.field(state, "u"))  # shape (Nt, Nx, Ny)

    # Build spatial grid once
    x = np.linspace(0, 1, args.Nx)
    y = np.linspace(0, 1, args.Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # build a time grid so we can't overshoot
    t_grid = np.linspace(0.0, 1.0, args.Nt)        # ← NEW

    # Times to plot
    times = np.arange(0.0, 1.01, 0.2)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, t in zip(axes, times):
        # find the nearest stored time index
        idx = np.argmin(np.abs(t_grid - t))       # ← CHANGED
        im = ax.pcolormesh(X, Y, u_ref[idx], shading="auto")
        ax.set_title(f"t = {t:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Direct (ODIL) solution snapshots")
    plt.savefig("direct_u_snapshots.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
