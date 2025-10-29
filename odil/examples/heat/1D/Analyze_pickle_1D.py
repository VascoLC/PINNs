#!/usr/bin/env python3
# analyze_heat_pickle.py

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.interpolate import griddata

try:
    import pandas as pd  # optional
except ImportError:
    pd = None


def load_pickle(path):
    """
    Load pickle with a couple of fallbacks for Python/NumPy compatibility.
    """
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    if not isinstance(data, dict):
        try:
            data = dict(data)
        except Exception:
            raise TypeError("Pickle does not contain a dict-like object.")
    return data


def summarize(data):
    print("\n=== Pickle contents summary ===")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"{k:>12}: ndarray, shape={v.shape}, dtype={v.dtype}, "
                  f"min={np.nanmin(v):.6g}, max={np.nanmax(v):.6g}")
        else:
            try:
                info = f"type={type(v).__name__}"
                if hasattr(v, "shape"):
                    info += f", shape={getattr(v, 'shape')}"
                if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                    info += f", len={len(v)}"
                print(f"{k:>12}: {info}")
            except Exception:
                print(f"{k:>12}: {type(v).__name__}")
    print("================================\n")


def default_axes_from_grid(u):
    """
    Build default t and x axes in [0,1] for a (Nt, Nx) field.
    """
    Nt, Nx = u.shape
    t = np.linspace(0.0, 1.0, Nt)
    x = np.linspace(0.0, 1.0, Nx)
    return t, x


def plot_state_u(u, t=None, x=None, title="Temperature Field", vmin=0.0, vmax=1.0,
                 cmap="YlOrBr", transpose=True, overlay_imp_points=None, savepath=None, show=False):
    # This function remains the same, used as a fallback for a single plot.
    if t is None or x is None:
        t, x = default_axes_from_grid(u)
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(6.4, 4.8))
    img = u
    plt.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cmap,
               vmin=vmin, vmax=vmax, interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label("U(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    if overlay_imp_points is not None and overlay_imp_points.size:
        tt = overlay_imp_points[:, 0]
        xx = overlay_imp_points[:, 1]
        plt.scatter(xx, tt, s=6, edgecolors="none", facecolors="k", alpha=0.8, zorder=10)
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"Saved heatmap to: {savepath}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_two_fields(u1, u2, t=None, x=None, title1="Plot 1", title2="Plot 2",
                    vmin=0.0, vmax=1.0, cmap="YlOrBr", savepath=None, show=False):
    # Renamed from plot_side_by_side for clarity
    if t is None or x is None:
        t, x = default_axes_from_grid(u1)
    extent = [x.min(), x.max(), t.min(), t.max()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(u1, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_title(title1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    im = axes[1].imshow(u2, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_title(title2)
    axes[1].set_xlabel("x")
    axes[1].tick_params(axis='y', labelleft=False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="U(x,t)")
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"Saved side-by-side plot to: {savepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_three_fields(u1, u2, u3, t=None, x=None, title1="Plot 1", title2="Plot 2", title3="Plot 3",
                      vmin=0.0, vmax=1.0, cmap="YlOrBr", savepath=None, show=False, overlay_points=None):
    """
    Plots three heatmaps side-by-side with a single shared colorbar,
    and optionally overlays scatter points on the first plot.
    """
    if t is None or x is None:
        t, x = default_axes_from_grid(u1)
    extent = [x.min(), x.max(), t.min(), t.max()]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1 (Left) with Optional Overlay ---
    axes[0].imshow(u1, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_title(title1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    # --- NEW: Overlay the imposed points on the first plot only ---
    if overlay_points is not None and overlay_points.size:
        tt = overlay_points[:, 0]
        xx = overlay_points[:, 1]
        axes[0].scatter(xx, tt, s=12, c='black', edgecolors='black',
                        linewidths=0.1, alpha=0.9, zorder=10)
        axes[0].legend() # Show legend on the first plot

    # --- Plot 2 (Middle) ---
    axes[1].imshow(u2, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_title(title2)
    axes[1].set_xlabel("x")
    axes[1].tick_params(axis='y', labelleft=False)

    # --- Plot 3 (Right) ---
    im = axes[2].imshow(u3, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[2].set_title(title3)
    axes[2].set_xlabel("x")
    axes[2].tick_params(axis='y', labelleft=False)

    # Shared Colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label="U(x,t)", shrink=0.8)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"Saved 3-way plot to: {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_k_vs_u(ref_uk=None, k=None, ref_k=None, pinn_u=None, pinn_k=None,
                  title="Conductivity k(u)", savepath=None, show=False):
    """
    Plots k vs u for reference, inferred (pickle), and PINN (csv) data.
    """
    if ref_uk is None and pinn_u is None:
        print("No k(u) data to plot.")
        return

    plt.figure(figsize=(6.0, 4.5))

    # Plot inferred k(u) from the pickle file
    if k is not None and ref_uk is not None:
        plt.plot(ref_uk, k, label="ODIL: Inferred k(u)", linewidth=2, zorder=10, color='red')

    # Plot reference k(u) from the pickle file
    if ref_k is not None and ref_uk is not None:
        plt.plot(ref_uk, ref_k, label="Reference k(u)", linewidth=2, linestyle='--', color='black', zorder=5)

    # --- NEW: Plot k(u) from the PINN CSV file ---
    if pinn_u is not None and pinn_k is not None:
        # Sort the values by 'u' to ensure a clean line plot
        sort_indices = np.argsort(pinn_u)
        u_pinn_sorted = pinn_u[sort_indices]
        k_pinn_sorted = pinn_k[sort_indices]
        plt.plot(u_pinn_sorted, k_pinn_sorted, label="PINN: Inferred k(u)",
                 linewidth=2.0, linestyle='dashdot', color='blue', zorder=8)

    plt.xlabel("U(x,t)")
    plt.ylabel("k(u)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
        print(f"Saved k(u) plot to: {savepath}")

    if show:
        plt.show()
    else:
        plt.close()


def export_imposed_csv(data, out_csv):
    if pd is None:
        print("pandas not installed; cannot export CSV.")
        return
    missing = [k for k in ("imp_points", "imp_indices", "imp_u") if k not in data]
    if missing:
        print(f"Cannot export imposed CSV: missing {missing}")
        return
    imp_points = np.asarray(data["imp_points"])
    imp_indices = np.asarray(data["imp_indices"]).astype(int)
    imp_u_grid = np.asarray(data["imp_u"])
    Nt, Nx = imp_u_grid.shape
    imp_rows = imp_indices // Nx
    imp_cols = imp_indices % Nx
    u_vals = imp_u_grid[imp_rows, imp_cols]
    if len(imp_points) != len(u_vals):
        print("Warning: imp_points length mismatch; exporting min length.")
    N = min(len(imp_points), len(u_vals))
    df = pd.DataFrame({"t": imp_points[:N, 0], "x": imp_points[:N, 1], "u": u_vals[:N]})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Exported imposed (t,x,u) data to: {out_csv}")


def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Analyze and plot 1D inverse heat pickle dump.")
    ap.add_argument("--pickle_path", type=str, default="C:\PINNs_Git\PINNs\odil\examples\heat\out_heat_inverse_1D_adam_ODIL/data_00030.pickle",help="Path to data_XXXXX.pickle")
    ap.add_argument("--save-dir", type=str, default="plots", help="Directory to save plots")
    ap.add_argument("--vmin", type=float, default=0.0, help="Heatmap min")
    ap.add_argument("--vmax", type=float, default=1.0, help="Heatmap max")
    ap.add_argument("--cmap", type=str, default="YlOrBr", help="Matplotlib colormap for u")
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    ap.add_argument("--pinn-csv", type=str, default="C:\PINNs_Git\PINNs\Heat_inverse\Heat_inverse_1D_5Batch_Nk/Heat_Inverse_1D_solution_Batching_NoK.csv", help="Path to PINN results in CSV format (t,x,u,k)")
    ap.add_argument("--no-transpose", action="store_true", help="This flag is no longer needed.")

    args = ap.parse_args()

    pkl_path = Path(args.pickle_path)
    if not pkl_path.exists():
        print(f"ERROR: Pickle file not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    data = load_pickle(pkl_path)
    summarize(data)

    df_pinn = None
    if args.pinn_csv and Path(args.pinn_csv).exists():
        print(f"Loading PINN data from: {args.pinn_csv}")
        df_pinn = pd.read_csv(args.pinn_csv)

    if "state_u" in data:
        u_state = np.asarray(data["state_u"])
        u_ref = data.get("ref_u", None)

        if df_pinn is not None and u_ref is not None:
            print("Found PINN CSV and ref_u. Generating 3-way comparison plot.")
            u_ref = np.asarray(u_ref)
            pinn_points_coords = df_pinn[['t', 'x']].values
            pinn_values = df_pinn['u'].values
            Nt, Nx = u_state.shape
            grid_t, grid_x = np.mgrid[0:1:Nt*1j, 0:1:Nx*1j]
            u_pinn = griddata(pinn_points_coords, pinn_values, (grid_t, grid_x), method='cubic')

            # --- NEW: Get the imposed points from the pickle data ---
            imp_points = data.get("imp_points", None)

            save_path = Path(args.save_dir) / (pkl_path.stem + "_3way_comparison.png")
            plot_three_fields(
                u_ref, u_state, u_pinn,
                title1="Reference",
                title2="ODIL Famework",
                title3="PINN Framework",
                vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
                savepath=str(save_path), show=args.show,
                overlay_points=imp_points 
            )

        elif u_ref is not None:
            print("Found 'ref_u' field. Generating side-by-side plot.")
            u_ref = np.asarray(u_ref)
            save_path = Path(args.save_dir) / (pkl_path.stem + "_comparison.png")
            plot_two_fields(
                u_ref, u_state,
                title1="Reference (ref_u)", title2="Inferred (state_u)",
                vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
                savepath=str(save_path), show=args.show
            )
        else:
            print("Did not find 'ref_u' or PINN data. Generating single plot.")
            save_path = Path(args.save_dir) / (pkl_path.stem + "_state_u.png")
            plot_state_u(u_state, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
                         overlay_imp_points=data.get("imp_points", None),
                         savepath=str(save_path), show=args.show)

    pinn_u_data = None
    pinn_k_data = None
    if df_pinn is not None and 'u' in df_pinn.columns and 'k' in df_pinn.columns:
        print("Found 'u' and 'k' columns in CSV for k(u) plot.")
        pinn_u_data = df_pinn['u'].values
        pinn_k_data = df_pinn['k'].values
    
    ref_uk = data.get("ref_uk", None)
    k = data.get("k", None)
    ref_k = data.get("ref_k", None)

    if ref_uk is not None or pinn_u_data is not None:
        save_k = Path(args.save_dir) / (pkl_path.stem + "_k_of_u.png")
        plot_k_vs_u(
            ref_uk=np.asarray(ref_uk) if ref_uk is not None else None,
            k=np.asarray(k) if k is not None else None,
            ref_k=np.asarray(ref_k) if ref_k is not None else None,
            pinn_u=pinn_u_data,
            pinn_k=pinn_k_data,
            title="",
            savepath=str(save_k),
            show=args.show,
        )

# This block is required to run the script
if __name__ == "__main__":
    main()