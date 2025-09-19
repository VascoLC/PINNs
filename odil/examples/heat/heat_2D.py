#!/usr/bin/env python3

import argparse
import numpy as np
import pickle
import os
import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
from odil.runtime import tf


def get_init_u(t, x, y=None, mod=np):
    """Initial condition - Gaussian profile in x, constant in y"""
    def f(z):
        return mod.exp(-((z - 0.5) ** 2) * 50)  # Wider Gaussian

    ux = f(x) - mod.exp(-12.5) # shape (Nx,)

    if y is None:
        return ux
    else:
        return mod.reshape(ux, [-1, 1]) * mod.ones((1, y.shape[0]), dtype=ux.dtype)


def get_ref_k(u, mod=np):
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))


def get_anneal_factor(epoch, period):
    return 0.5 ** (epoch / period) if period else 1


def transform_k(knet, mod, kmax):
    return mod.sigmoid(knet) * kmax


def operator_odil(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args
    dt, dx, dy = ctx.step()
    it, ix, iy = ctx.indices()
    nt, nx, ny = ctx.size()
    epoch = ctx.tracers["epoch"]

    def stencil_var(key, frozen=False):
        if not args.keep_frozen:
            frozen = False
        return [
            [  # x-direction stencil
                ctx.field(key, 0, 0, 0, frozen=frozen),
                ctx.field(key, 0, -1, 0, frozen=frozen),
                ctx.field(key, 0, 1, 0, frozen=frozen)
            ],
            [  # y-direction stencil
                ctx.field(key, 0, 0, -1, frozen=frozen),
                ctx.field(key, 0, 0, 0, frozen=frozen),
                ctx.field(key, 0, 0, 1, frozen=frozen)
            ],
            [  # time stencil
                ctx.field(key, -1, 0, 0, frozen=frozen),
                ctx.field(key, -1, -1, 0, frozen=frozen),
                ctx.field(key, -1, 1, 0, frozen=frozen)
            ]
        ]

    def apply_bc_u(st):
        # Apply boundary conditions
        # x-direction: Dirichlet (u=0)
        uc, um, up = st[0]
        st[0][1] = mod.where(ix == 0, -uc, um)     # Left boundary
        st[0][2] = mod.where(ix == nx-1, -uc, up)  # Right boundary
        
        # y-direction: Neumann (du/dy=0)
        uc, um, up = st[1]
        st[1][1] = mod.where(iy == 0, uc, um)     # Bottom boundary
        st[1][2] = mod.where(iy == ny-1, uc, up)   # Top boundary
        return st

    # Get stencils
    u_st = stencil_var("u")
    apply_bc_u(u_st)
    uf_st = stencil_var("u", frozen=True)
    apply_bc_u(uf_st)

    # Time derivative
    u_t = (u_st[0][0] - u_st[2][0]) / dt

    # Spatial derivatives - centered differences
    u_xm = (u_st[0][0] + u_st[2][0] - u_st[0][1] - u_st[2][1]) / (2*dx)
    u_xp = (u_st[0][2] + u_st[2][2] - u_st[0][0] - u_st[2][0]) / (2*dx)
    u_ym = (u_st[1][0] + u_st[2][0] - u_st[1][1] - u_st[2][1]) / (2*dy)
    u_yp = (u_st[1][2] + u_st[2][2] - u_st[1][0] - u_st[2][0]) / (2*dy)

    # Interpolate to half-faces
    ufxmh = ((uf_st[0][0] + uf_st[2][0]) + (uf_st[0][1] + uf_st[2][1])) * 0.25
    ufxph = ((uf_st[0][2] + uf_st[2][2]) + (uf_st[0][0] + uf_st[2][0])) * 0.25
    ufymh = ((uf_st[1][0] + uf_st[2][0]) + (uf_st[1][1] + uf_st[2][1])) * 0.25
    ufyph = ((uf_st[1][2] + uf_st[2][2]) + (uf_st[1][0] + uf_st[2][0])) * 0.25

    # Conductivity - use interpolated face values
    if args.infer_k:
        kmx = transform_k(ctx.neural_net("k_net")(ufxmh)[0], mod, args.kmax)
        kpx = transform_k(ctx.neural_net("k_net")(ufxph)[0], mod, args.kmax)
        kmy = transform_k(ctx.neural_net("k_net")(ufymh)[0], mod, args.kmax)
        kpy = transform_k(ctx.neural_net("k_net")(ufyph)[0], mod, args.kmax)
    else:
        kmx = get_ref_k(ufxmh, mod)
        kpx = get_ref_k(ufxph, mod)
        kmy = get_ref_k(ufymh, mod)
        kpy = get_ref_k(ufyph, mod)

    # Flux divergence
    qx = u_xp * kpx - u_xm * kmx
    qy = u_yp * kpy - u_ym * kmy
    q_div = qx/dx + qy/dy

    # PDE residual
    fu = u_t - q_div
    fu = mod.where(it == 0, ctx.cast(0), fu)  # Skip at t=0
    res = [("fu", fu)]

    # Initial condition
    if args.keep_init:
        u_here = ctx.field("u", 0, 0, 0)
        ic = u_here - mod.cast(extra.u0, u_here.dtype)
        init_res = mod.where(it == 0, ic, ctx.cast(0))
        res.append(("init", init_res))

    # Imposed points
    if extra.imp_size:
        # bring everything into TF-land
        u_cur = u_st[0][0]  # the current u field, a tf.Tensor
        dtype = u_cur.dtype

        # cast the numpy arrays/scalars to TF tensors
        mask   = ctx.cast(extra.imp_mask, dtype)       # shape (Nt,Nx,Ny)
        imp_u  = ctx.cast(extra.imp_u,   dtype)       # same shape
        # compute k_imp as a TF scalar
        k_imp_val = args.kimp * (np.prod(ctx.size()) / extra.imp_size) ** 0.5
        k_imp = ctx.cast(k_imp_val, dtype)

        # now do the residual entirely in TF
        fuimp = mask * (u_cur - imp_u) * k_imp
        res.append(("imp", fuimp))

    if args.kwreg and args.infer_k:
        domain = ctx.domain
        ww = domain.arrays_from_field(ctx.state.fields["k_net"])
        ww = mod.concatenate([mod.flatten(w) for w in ww], axis=0)
        k_ = args.kwreg * get_anneal_factor(epoch, args.kwregdecay)
        k = mod.cast(k_,ww.dtype)
        res += [("wreg", (mod.stop_gradient(ww) - ww) * k)]

    return res

def operator_pinn(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args

    # Inner points.
    inputs = [mod.constant(extra.t_inner), mod.constant(extra.x_inner)]
    (u,) = ctx.neural_net("u_net")(*inputs)

    def grad(f, *deriv):
        for i in range(len(deriv)):
            for _ in range(deriv[i]):
                f = tf.gradients(f, inputs[i])[0]
        return f

    u_t = grad(u, 1, 0)
    u_x = grad(u, 0, 1)

    # Conductivity.
    if args.infer_k:
        k = transform_k(ctx.neural_net("k_net")(u)[0], mod, args.kmax)
    else:
        k = get_ref_k(u, mod=mod)
    q = k * u_x
    q_x = grad(q, 0, 1)

    res = []

    # Heat equation.
    fu = u_t - q_x
    res += [("eqn", fu)]

    # Boundary conditions.
    (u_net_bound,) = ctx.neural_net("u_net")(extra.t_bound, extra.x_bound)
    fb = u_net_bound - extra.u_bound
    res += [("bound", fb)]

    # Initial conditions.
    if args.keep_init:
        (u_net_init,) = ctx.neural_net("u_net")(extra.t_init, extra.x_init)
        fi = u_net_init - extra.u_init
        res += [("init", fi)]

    # Imposed points.
    if extra.imp_size:
        imp_t, imp_x = extra.imp_points.T
        (u_net_imp,) = ctx.neural_net("u_net")(imp_t, imp_x)
        imp_indices = mod.reshape(extra.imp_indices, [-1, 1])
        u_imp = mod.gather_nd(mod.flatten(extra.imp_u), imp_indices)
        fimp = (u_net_imp - u_imp) * args.kimp
        res += [("imp", fimp)]

    return res


def get_imposed_indices(domain, args):
    """
    Returns a 1D array of flat indices (0…Nt*Nx*Ny−1) 
    of the points where you want to impose extra data.
    """
    size = np.prod(domain.cshape)

    # domain.points() under TF gives EagerTensors – convert them to numpy
    t_full, x_full, y_full = domain.points()
    t_flat = np.array(t_full).ravel()    # now a numpy array, shape (size,)

    all_idx = np.arange(size)

    rng = np.random.default_rng(args.seed)
    if args.imposed == "random":
        nimp = min(args.nimp, size)
        return rng.choice(all_idx, size=nimp, replace=False)

    elif args.imposed == "stripe":
        # pick those cells whose t is within 1/6 of 0.5
        cand = all_idx[np.abs(t_flat - 0.5) < (1/6)]
        nimp = min(args.nimp, cand.size)
        return rng.choice(cand, size=nimp, replace=False)

    else:  # "none"
        return np.array([], dtype=int)


def get_imposed_mask(args, domain):
    """
    Builds both the mask (shape cshape) and the list of flat indices.
    """
    size = np.prod(domain.cshape)
    imp_i = get_imposed_indices(domain, args)   # flat indices

    # build a flat mask, then reshape
    mask = np.zeros(size, dtype=np.float32)
    mask[imp_i] = 1.0
    mask = mask.reshape(domain.cshape)          # back to (Nt, Nx, Ny)

    # if you also want the physical coords of imposed points:
    t_full, x_full, y_full = domain.points()
    t_flat = np.array(t_full).ravel()
    x_flat = np.array(x_full).ravel()
    y_flat = np.array(y_full).ravel()
    coords = np.stack([
        t_flat[imp_i],
        x_flat[imp_i],
        y_flat[imp_i],
    ], axis=1)

    return mask, coords, imp_i



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--Nt", type=int, default=64, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=64, help="Grid size in y")
    parser.add_argument('--ndim',
                    type=int,
                    choices=[1, 2, 3, 4, 5, 6],
                    default=3,
                    help="Space dimension")
    parser.add_argument("--Nci", type=int, default=4096, help="Number of collocation points inside domain")
    parser.add_argument("--Ncb", type=int, default=128, help="Number of collocation points on each boundary")
    parser.add_argument(
        "--arch_u", type=int, nargs="*", default=[10, 10], help="Network architecture for temperature in PINN"
    )
    parser.add_argument(
        "--arch_k", type=int, nargs="*", default=[20,20,20], help="Network architecture for inferred conductivity"
    )
    parser.add_argument("--solver", type=str, choices=("pinn", "odil"), default="odil", help="Grid size in x")
    parser.add_argument("--infer_k", type=int, default=1, help="Infer conductivity")
    parser.add_argument("--kxreg", type=float, default=0, help="Space regularization weight")
    parser.add_argument("--kxregdecay", type=float, default=0, help="Decay period of kxreg")
    parser.add_argument("--ktreg", type=float, default=0, help="Time regularization weight")
    parser.add_argument("--ktregdecay", type=float, default=0, help="Decay period of ktreg")
    parser.add_argument("--kwreg", type=float, default=0.1, help="Regularization of neural network weights")
    parser.add_argument("--kwregdecay", type=float, default=0, help="Decay period of kwreg")
    parser.add_argument("--kimp", type=float, default=5, help="Weight of imposed points")
    parser.add_argument("--keep_frozen", type=int, default=1, help="Respect frozen attribute for fields")
    parser.add_argument("--keep_init", type=int, default=1, help="Impose initial conditions")
    parser.add_argument("--ref_path",type=str, default="out_heat2D_direct/ref_solution.pickle",help="Path to reference solution *.pickle")
    parser.add_argument(
        "--imposed",
        type=str,
        choices=["random", "stripe", "none"],
        default="random",
        help="Set of points for imposed solution",
    )
    parser.add_argument("--nimp", type=int, default=500, help="Number of points for imposed=random")
    parser.add_argument("--noise", type=float, default=0, help="Magnitude of perturbation of reference solution")
    parser.add_argument("--kmax", type=float, default=0.1, help="Maximum value of conductivity")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir="out_heat2D_inverse")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=0)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(plot_every=200, report_every=200, history_full=5, history_every=200, frames=5)
    return parser.parse_args()


@tf.function()
def eval_u_net(domain, net, arrays):
    domain.arrays_to_field(arrays, net)
    tt, xx = domain.points()
    (net_u,) = odil.core.eval_neural_net(net, [tt, xx], domain.mod)
    return net_u


def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title0 = "u epoch={:}".format(epoch) if args.plot_title else None
    title1 = "k epoch={:}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "k_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1)

    if args.solver == "odil":
        state_u = domain.field(state, "u")
    elif args.solver == "pinn":
        net = state.fields["u_net"]
        arrays = domain.arrays_from_field(net)
        state_u = eval_u_net(domain, net, arrays)
    state_u = np.array(state_u)

    t_vals = np.arange(0.0, 1.01, 0.2)
    t_grid = domain.points_1d()[0]  # 1D time grid
    x1, y1 = domain.points_1d()[1:]  # 1D spatial grids
    ref_u = np.array(extra.ref_u)
    vmin = min(state_u.min(), ref_u.min())
    vmax = max(state_u.max(), ref_u.max())


    for tval in t_vals:
        t_index = np.argmin(np.abs(t_grid - tval))
        u_pred = state_u[t_index]
        u_ref = ref_u[t_index]
        u_diff = np.abs(u_pred - u_ref)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

        for ax, data, title in zip(
            axes,
            [u_ref, u_pred, u_diff],
            [f"Reference\n$t={tval:.1f}$", f"Prediction\n$t={tval:.1f}$", "Abs. Error"]
        ):
            im = ax.imshow(data.T, extent=[x1[0], x1[-1], y1[0], y1[-1]],
                        origin="lower", aspect="auto", cmap="viridis" if title != "Abs. Error" else "Reds",
                        vmin=vmin if title != "Abs. Error" else 0,
                        vmax=vmax if title != "Abs. Error" else u_diff.max())
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)

        fname = f"compare_t{int(tval * 10):02d}.{args.plotext}"  # e.g., compare_t00.png
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    # Plot conductivity.
    fig, ax = plt.subplots(figsize=(1.7, 1.5))
    ref_uk = extra.ref_uk
    ref_k = get_ref_k(ref_uk)

    if args.infer_k:
         (k_net_out,) = domain.neural_net(state, "k_net")(ref_uk)
         k_pred = transform_k(k_net_out, mod, args.kmax)
         k_pred = np.array(k_pred)
    else:
        k = None
    if k_pred is not None:
         ax.plot(ref_uk, k_pred, label="learned", zorder=10)

    ax.plot(ref_uk, ref_k,    c="C2", lw=1.5, label="true", zorder=1)
    ax.set_xlabel("u")
    ax.set_ylabel("k")
    ax.set_ylim(0, 0.03)
    ax.set_title(title1)
    ax.legend(loc="best")
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)

    if frame == args.frames:
        state_to_save = odil.State(fields={"u": odil.Field(array=state_u)})
        state_to_save = domain.init_state(state_to_save)

        odil.core.checkpoint_save(domain, state_to_save, "ref_solution.pickle")


    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        d = dict()
        d["state_u"] = state_u
        d["ref_u"] = extra.ref_u  # Reference without noise.
        d["imp_u"] = extra.imp_u  # Reference with noise.
        d["ref_uk"] = ref_uk
        if args.infer_k:
            d["k_pred"] = k_pred
        else:
            d["k_pred"] = None
        d["ref_k"] = ref_k
        d["imp_indices"] = extra.imp_indices
        d["imp_points"] = extra.imp_points
        d = odil.core.struct_to_numpy(mod, d)
        with open(path, "wb") as f:
            pickle.dump(d, f)


def get_error(domain, extra, state, key):
    args = extra.args
    mod = domain.mod
    if key == "u":
        if args.solver == "odil":
            state_u = np.array(domain.field(state, key))
        elif args.solver == "pinn":
            net = state.fields["u_net"]
            arrays = domain.arrays_from_field(net)
            state_u = eval_u_net(domain, net, arrays)
            state_u = np.array(eval_u_net(domain, net, arrays))
        ref_u = extra.ref_u
        return np.sqrt(np.mean((state_u - ref_u) ** 2))
    elif key == "k" and args.infer_k:
        (k,) = domain.neural_net(state, "k_net")(extra.ref_uk)
        k = transform_k(k, mod, args.kmax)
        max_k = extra.ref_k.max()
        return np.sqrt(np.mean((k - extra.ref_k) ** 2)) / max_k
    return None


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in ["u", "k"]:
        error = get_error(domain, extra, state, key)
        if error is not None:
            history.append("error_" + key, error)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    extra = problem.extra
    res = dict()
    for key in ["u", "k"]:
        error = get_error(domain, extra, state, key)
        if error is not None:
            res[key] = error
    printlog("error: " + ", ".join("{}:{:.5g}".format(*item) for item in res.items()))


def load_fields_interp(path, keys, domain):
    """
    Loads fields from file `path` and interpolates them to shape `domain.cshape`.

    keys: `list` of `str`
        Keys of fields to load.
    """
    from scipy.interpolate import RegularGridInterpolator

    src_state = odil.State(fields={key: odil.Field() for key in keys})
    state = odil.State(fields={key: odil.Field() for key in keys})
    odil.core.checkpoint_load(domain, src_state, path)

    tgt_shape = domain.cshape
    tgt_points = domain.points()
    tgt_coords = [np.array(p).flatten() for p in tgt_points]  # [Nt, Nx, Ny]

    for key in keys:
        src_u = src_state.fields[key].array
        if src_u.shape == tgt_shape:
            state.fields[key].array = src_u  # No interpolation needed
        else:
            # Assume source is also 3D (Nt, Nx, Ny)
            Nt_src, Nx_src, Ny_src = src_u.shape
            t_src = np.linspace(domain.lower[0], domain.upper[0], Nt_src)
            x_src = np.linspace(domain.lower[1], domain.upper[1], Nx_src)
            y_src = np.linspace(domain.lower[2], domain.upper[2], Ny_src)

            interpolator = RegularGridInterpolator(
                (t_src, x_src, y_src), src_u, bounds_error=False, fill_value=None
            )

            # Create meshgrid of target points
            t_vals, x_vals, y_vals = tgt_points
            pts = np.stack([t_vals.flatten(), x_vals.flatten(), y_vals.flatten()], axis=-1)
            interp_vals = interpolator(pts).reshape(tgt_shape)
            state.fields[key].array = interp_vals

    return state


def make_problem(args):
    import numpy as np
    import argparse
    import odil
    from odil import printlog
    from odil.core import checkpoint_load
    from scipy.interpolate import RegularGridInterpolator

    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(
        cshape=(args.Nt, args.Nx, args.Ny),
        dimnames=("t", "x", "y"),
        multigrid=args.multigrid,
        dtype=dtype
    )
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)
    mod = domain.mod

    # helper to load and interpolate a saved pickle into our domain
    def load_ref(path):
        src = odil.State(fields={"u": odil.Field()})
        checkpoint_load(domain, src, path)
        data = np.array(src.fields["u"].array)
        # if shapes match, fine, otherwise you'd interpolate here…
        return data

    # 1) Build the initial bump at t=0 (we’ll need it for plotting/reference in direct mode)
    _, x1, y1 = domain.points_1d()
    u0 = get_init_u(None, x1, y1, mod)

    # 2) Reference data / imposed data
    if args.infer_k == 1:
        # — inverse run: load the direct solution you saved previously
        printlog("INVERSE mode: loading ref_solution.pickle")
        ref_u = load_ref(args.ref_path)
    else:
        # — direct run: no `ref_u` to load, ODIL will solve in-place
        #   but we still need something to pass into plot_func so it can compare
        #   (you could also bypass comparison in direct mode)
        printlog("DIRECT mode: no reference loaded")
        # here I just tile the initial bump in time so plot_func can compare
        Nt = args.Nt
        ref_u = np.stack([u0]*Nt, axis=0)

    # add noise (if requested) to create your “data” for inverse
    imp_u = ref_u.copy()
    if args.noise:
        rng = np.random.default_rng(args.seed)
        imp_u += rng.normal(0, args.noise, size=imp_u.shape)

    imp_mask, imp_points, imp_indices = get_imposed_mask(args, domain)
    imp_size = len(imp_points)
    # (you already write imposed.csv later)

    # 3) Pack everything into extra
    extra = argparse.Namespace()
    extra.args        = args
    extra.ref_u       = ref_u
    extra.imp_u       = imp_u
    extra.imp_mask    = imp_mask
    extra.imp_points  = imp_points
    extra.imp_indices = imp_indices
    extra.imp_size    = len(imp_indices)
    extra.ref_uk      = np.linspace(0,1,200).astype(domain.dtype)
    extra.ref_k       = get_ref_k(extra.ref_uk)
    extra.u0          = u0
    extra.epoch       = mod.variable(domain.cast(0))

    # PINN‐only extras
    if args.solver == "pinn":
        t_i, x_i, y_i = domain.random_inner(args.Nci)
        t_b0,x_b0,y_b0 = domain.random_boundary(1,0,args.Ncb)
        t_b1,x_b1,y_b1 = domain.random_boundary(1,1,args.Ncb)
        t_bound = np.hstack((t_b0,t_b1))
        x_bound = np.hstack((x_b0,x_b1))
        t_init, x_init, y_init = domain.random_boundary(0,0,args.Ncb)
        extra.t_inner, extra.x_inner, extra.y_inner = t_i, x_i, y_i
        extra.t_bound, extra.x_bound, extra.y_bound = t_bound, x_bound, np.hstack((y_b0,y_b1))
        extra.t_init, extra.x_init, extra.y_init = t_init, x_init, y_init
        extra.u_init = get_init_u(t_init, x_init, mod)
        extra.u_bound = get_init_u(t_bound, x_bound, mod)

    # 4) Build the State & Problem (same for both modes)
    state = odil.State()
    if args.solver == "odil":
        state.fields["u"] = np.zeros(domain.cshape)
        operator = operator_odil
    else:
        state.fields["u_net"] = domain.make_neural_net([2] + args.arch_u + [1])
        operator = operator_pinn

    if args.infer_k:
        state.fields["k_net"] = domain.make_neural_net([1] + args.arch_k + [1])

    state   = domain.init_state(state)
    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args, relpath_args=["checkpoint", "checkpoint_train", "ref_path"])
    problem, state = make_problem(args)
    callback = odil.make_callback(
        problem, args, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open("done", "w") as f:
        pass
    
if __name__ == "__main__":
    main()