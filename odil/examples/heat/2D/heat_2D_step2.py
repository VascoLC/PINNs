#!/usr/bin/env python3

import argparse
import numpy as np
import pickle

import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
from odil.runtime import tf


def get_init_u(t, x, y=None, mod=np):
    # Gaussian.
    def f(z):
        return mod.exp(-((z - 0.5) ** 2) * 50)
    zero_val = mod.cast(0.0, x.dtype)
    return f(x) - f(zero_val)


def get_ref_k(u, mod=np):
    # Gaussian.
    return 0.02 * (mod.exp(-((u - 0.5) ** 2) * 20))


def get_anneal_factor(epoch, period):
    return 0.5 ** (epoch / period) if period else 1


def transform_k(knet, mod, kmax):
    return mod.sigmoid(knet) * kmax

def get_tri_bc(x, x_peak=0.5, width=0.5, amplitude=1.0, mod=np):

    half_width = width / 2.0
    norm_dist = mod.abs(x - x_peak) / half_width
    value = amplitude * (1.0 - norm_dist)
    
    return mod.maximum(mod.cast(0.0, x.dtype), value)

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
            [
                ctx.field(key, 0, 0, 0, frozen=frozen),
                ctx.field(key, 0, -1, 0, frozen=frozen),
                ctx.field(key, 0, 1, 0, frozen=frozen),
                ctx.field(key, 0, 0, -1, frozen=frozen),
                ctx.field(key, 0, 0, 1, frozen=frozen),
            ],
            [
                ctx.field(key, -1, 0, 0, frozen=frozen),
                ctx.field(key, -1, -1, 0, frozen=frozen),
                ctx.field(key, -1, 1, 0, frozen=frozen),
                ctx.field(key, -1, 0, -1, frozen=frozen),
                ctx.field(key, -1, 0, 1, frozen=frozen),
            ],
        ]


    def apply_bc_u(st):
        # Apply boundary conditions by extrapolation to halo cells.
        if args.keep_init:
            # Initial conditions, linear extrapolation.
            u0 = extra.init_u
            q0 = [u0, mod.roll(u0, 1, axis=0), mod.roll(u0, -1, axis=0),
                  mod.roll(u0, 1, axis=1), mod.roll(u0, -1, axis=1)]
            extrap = odil.core.extrap_linear
            q, qm = st
            for i in range(5):
                qm[i] = mod.where(it == 0, extrap(q[i], q0[i][None, :]), qm[i])
        # Zero Dirichlet conditions, quadratic extrapolation.
        extrap = odil.core.extrap_quadh

        y_bc1 = ctx.domain.points()[2] 
        tri = mod.where(y_bc1 <= 0.5,y_bc1 / 0.5,(1 - y_bc1) / 0.5)
        tri = mod.maximum(mod.cast(0.0, y_bc1.dtype), tri) 

        for q in st:
            # If iy ==0, substitute hallo cell q[3] by q[0] for zero gradient 
            q[1] = mod.where(ix == 0, extrap(q[2], q[0], 0), q[1])
            # Ghost point technique: for the BC value, one should average the interior point and hallo cell
            #q[2] = mod.where(ix == nx - 1, extrap(q[1], q[0], 0), q[2]) 
            q[2] = mod.where(ix == nx - 1, tri[None, :], q[2])
            q[3] = mod.where(iy == 0, q[0], q[3])
            q[4] = mod.where(iy == ny - 1, q[0], q[4])
        return st

    u_st = stencil_var("u")
    apply_bc_u(u_st)

    q, qm = u_st

    u_cen, u_w, u_e, u_s, u_n = q
    um_cen, um_w, um_e, um_s, um_n = qm

    u_xm = ((u_cen + um_cen) - (u_w + um_w)) / (2 * dx)
    u_xp = ((u_e + um_e) - (u_cen + um_cen)) / (2 * dx) 

    u_ym = ((u_cen + um_cen) - (u_s + um_s)) / (2 * dy) 
    u_yp = ((u_n + um_n) - (u_cen + um_cen)) / (2 * dy) 

    u_t = (u_cen - um_cen) / dt

    uf_st = stencil_var("u", frozen=True)
    apply_bc_u(uf_st)
    qf, qfm = uf_st
    uf_cen, uf_w, uf_e, uf_s, uf_n = qf
    ufm_cen, ufm_w, ufm_e, ufm_s, ufm_n = qfm

    ufxmh = ((uf_cen + ufm_cen) + (uf_w + ufm_w)) * 0.25 # West face
    ufxph = ((uf_e + ufm_e) + (uf_cen + ufm_cen)) * 0.25 # East face

    ufymh = ((uf_cen + ufm_cen) + (uf_s + ufm_s)) * 0.25 # South face
    ufyph = ((uf_n + ufm_n) + (uf_cen + ufm_cen)) * 0.25 # North face

    # Conductivity.
    if args.infer_k:
        k_net = ctx.neural_net("k_net")
        km_x = transform_k(k_net(ufxmh)[0], mod, args.kmax)
        kp_x = transform_k(k_net(ufxph)[0], mod, args.kmax)
        km_y = transform_k(k_net(ufymh)[0], mod, args.kmax) # ADDED
        kp_y = transform_k(k_net(ufyph)[0], mod, args.kmax) # ADDED
    else:
        km_x = get_ref_k(ufxmh, mod=mod)
        kp_x = get_ref_k(ufxph, mod=mod)
        km_y = get_ref_k(ufymh, mod=mod) # ADDED
        kp_y = get_ref_k(ufyph, mod=mod) # ADDED


    # Heat equation.
    # Fluxes in x-direction
    qm_x = u_xm * km_x
    qp_x = u_xp * kp_x
    # ADDED: Fluxes in y-direction
    qm_y = u_ym * km_y
    qp_y = u_yp * kp_y
    # Divergence of the flux    
    q_div_x = (qp_x - qm_x) / dx
    q_div_y = (qp_y - qm_y) / dy # ADDED

    
    dom = ctx.domain 
    t = ctx.cast(it) * dt
    x = dom.lower[1] + ctx.cast(ix) * dx

    if args.infer_s:
        points = ctx.points() 
        S = ctx.neural_net("s_net")(*list(points))
    else:
        A  = args.src_strength
        v  = args.src_speed
        x0 = args.src_x0

        xc = x - (x0 + v*t)
        half_w = args.src_width * 0.5

        S = A * ctx.cast(mod.abs(xc) <= half_w, dtype=u_t.dtype)

    # MODIFIED: Final residual includes both divergence terms
    #fu = u_t - (q_div_x + q_div_y) - S
    fu = u_t - (q_div_x + q_div_y)
    
    if not args.keep_init:
        fu = mod.where(it == 0, ctx.cast(0), fu)
    res = [("fu", fu)]

    if extra.imp_size:
        u = u_st[0]
        # Rescale weight to the total number of points.
        b = args.kimp * (np.prod(ctx.size()) / extra.imp_size) ** 0.5
        # imp_mask is a tensor of 0s and 1s (1s in the location of the imposed points, and 0s in the location of non-existent imposed points)
        fuimp = extra.imp_mask * (u_st[0][0] - extra.imp_u) * b
        res += [("imp", fuimp)]
        #(k_pred,) = ctx.domain.neural_net(ctx.state,"k_net")(extra.ref_uk)
        #k_pred = transform_k(k_pred, mod, args.kmax)
        #fimp_k = (k_pred - extra.ref_k) * 100
        #res += [("imp_k", fimp_k)]

    # Regularization.
    if args.kxreg:
        k_ = args.kxreg * get_anneal_factor(epoch, args.kxregdecay)
        u_x = (u_st[0][0] - u_st[0][1]) / dx
        u_x = mod.where(ix == 0, ctx.cast(0), u_x)
        k = mod.cast(k_,u_x.dtype)
        fxreg= u_x * k
        res += [("xreg", fxreg)]

    # ADDED: y-direction spatial regularization
    if args.kyreg: 
        k_ = args.kyreg * get_anneal_factor(epoch, args.kyregdecay)
        u_y = (u_st[0][0] - u_st[0][3]) / dy 
        u_y = mod.where(iy == 0, ctx.cast(0), u_y) 
        k = mod.cast(k_, u_y.dtype)
        fyreg = u_y * k
        res += [("yreg", fyreg)]

    if args.ktreg:
        k_ = args.ktreg * get_anneal_factor(epoch, args.ktregdecay)
        u_t = (u_st[0][0] - u_st[1][0]) / dt
        u_t = mod.where(it == 0, ctx.cast(0), u_t)
        k = mod.cast(k_,u_t.dtype)
        ftreg = u_t * k
        res += [("treg", ftreg)]

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


def get_imposed_indices(domain, args, iflat):
    iflat = np.array(iflat)
    rng = np.random.default_rng(args.seed)
    if args.imposed == "random":
        imp_i = iflat.flatten()
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == "stripe":
        imp_i = iflat.flatten()
        t = np.array(domain.points("t")).flatten()
        imp_i = imp_i[abs(t[imp_i] - 0.5) < 1 / 6]
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == "none":
        imp_i = []
    else:
        raise ValueError("Unknown imposed=" + args.imposed)
    return imp_i


def get_imposed_mask(args, domain):
    mod = domain.mod
    size = np.prod(domain.cshape)
    row = range(size)
    iflat = np.reshape(row, domain.cshape)
    imp_i = get_imposed_indices(domain, args, iflat)
    imp_i = np.unique(imp_i)
    mask = np.zeros(size)
    if len(imp_i):
        mask[imp_i] = 1
        points = [mod.flatten(domain.points(i)) for i in range(domain.ndim)]
        points = np.array(points)[:, imp_i].T
    else:
        points = np.zeros((0, domain.ndim))
    mask = mask.reshape(domain.cshape)
    return mask, points, imp_i


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--Nt", type=int, default=32, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=32, help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=32, help="Grid size in y")
    parser.add_argument("--Nci", type=int, default=4096, help="Number of collocation points inside domain")
    parser.add_argument("--Ncb", type=int, default=128, help="Number of collocation points on each boundary")
    parser.add_argument(
        "--arch_u", type=int, nargs="*", default=[10, 10], help="Network architecture for temperature in PINN"
    )
    parser.add_argument(
        "--arch_k", type=int, nargs="*", default=[5, 5], help="Network architecture for inferred conductivity"
    )
    parser.add_argument("--solver", type=str, choices=("pinn", "odil"), default="odil", help="Framework")
    parser.add_argument("--infer_k", type=int, default=1, help="Infer conductivity")
    parser.add_argument("--infer_s", type=int, default=0, help="Infer the source term S(t,x,y)")
    parser.add_argument(
        "--arch_s", type=int, nargs="*", default=[5, 5], help="Network architecture for inferred source")
    parser.add_argument("--kxreg", type=float, default=0, help="Space regularization weight")
    parser.add_argument("--kxregdecay", type=float, default=0, help="Decay period of kxreg")
    parser.add_argument("--kyreg", type=float, default=0, help="Space regularization weight")
    parser.add_argument("--kyregdecay", type=float, default=0, help="Decay period of kyreg")
    parser.add_argument("--ktreg", type=float, default=0, help="Time regularization weight")
    parser.add_argument("--ktregdecay", type=float, default=0, help="Decay period of ktreg")
    parser.add_argument("--kwreg", type=float, default=0, help="Regularization of neural network weights")
    parser.add_argument("--kwregdecay", type=float, default=0, help="Decay period of kwreg")
    parser.add_argument("--kimp", type=float, default=2, help="Weight of imposed points")
    parser.add_argument("--keep_frozen", type=int, default=1, help="Respect frozen attribute for fields")
    parser.add_argument("--keep_init", type=int, default=1, help="Impose initial conditions")
    parser.add_argument("--ref_path", type=str,default='Second_Step\out_heat_direct_Tribc/data_00020.pickle',
                         help="Path to reference solution *.pickle")
    parser.add_argument(
        "--imposed",
        type=str,
        choices=["random", "stripe", "none"],
        default="random",
        help="Set of points for imposed solution",
    )
    parser.add_argument("--nimp", type=int, default=250, help="Number of points for imposed=random")
    parser.add_argument("--noise", type=float, default=0, help="Magnitude of perturbation of reference solution")
    parser.add_argument("--kmax", type=float, default=0.1, help="Maximum value of conductivity")
    parser.add_argument("--src_strength", type=float, default=5.0, help="Strength (A) of the moving source")
    parser.add_argument("--src_speed", type=float, default=0.4, help="Horizontal speed (v) of the moving source")
    parser.add_argument("--src_x0", type=float, default=0.6, help="Initial x-position (x0) of the source center at t=0")
    parser.add_argument("--src_y0", type=float, default=0.5, help="Initial y-position (y0) of the source (unused)")
    parser.add_argument("--src_width", type=float, default=0.05, help="Width of the moving source bar")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir="Second_Step/out_heat_inverse_adam")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=0)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(plot_every=2500, report_every=100, history_full=10, history_every=100, frames=16)
    return parser.parse_args()

@tf.function()
def eval_u_net(domain, net, arrays):
    domain.arrays_to_field(arrays, net)
    tt, xx = domain.points()
    (net_u,) = odil.core.eval_neural_net(net, [tt, xx], domain.mod)
    return net_u

def plot_func(problem, state, epoch, frame, cbinfo=None):
    import matplotlib.pyplot as plt
    
    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    # --- Setup paths and titles ---
    title0 = f"Epochs={epoch}" if args.plot_title else None
    title1 = f"Epochs={epoch}" if args.plot_title else None
    path0 = f"u_{frame:05d}.{args.plotext}"
    path1 = f"k_{frame:05d}.{args.plotext}"
    printlog(f"Saving plots: {path0}, {path1}")

    # --- Get the solution field 'u' ---
    if args.solver == "odil":
        state_u = domain.field(state, "u")
    elif args.solver == "pinn":
        # Note: Ensure eval_u_net is updated for 3 inputs (t, x, y)
        net = state.fields["u_net"]
        arrays = domain.arrays_from_field(net)
        state_u = eval_u_net(domain, net, arrays)
    state_u = np.array(state_u) # Shape is (Nt, Nx, Ny)

    # --- Plot 1: Heat distribution u(t, x, y) at different time slices ---
    
    # Define how many time slices to display
    nslices = 5
    # Create a figure with a row of subplots
    fig, axes = plt.subplots(1, nslices, figsize=(nslices * 3, 3.5), sharey=True)
    fig.suptitle(title0)

    # Get the physical boundaries of the spatial domain
    t_coords, x_coords, y_coords = domain.points_1d()
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    # Select evenly spaced time indices to plot
    time_indices = np.linspace(0, state_u.shape[0] - 1, nslices, dtype=int)
    
    # Determine the time window for showing imposed points on each slice
    dt = domain.step()[0]
    time_window = dt * (len(t_coords) / nslices) / 2

    # Find global min/max for a consistent color scale
    umin = 0
    umax = 1

    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        t_val = t_coords[t_idx]
        
        # Extract the 2D spatial slice at this time
        u_slice = state_u[t_idx, :, :]
        
        # Plot the heatmap of the solution
        im = ax.imshow(u_slice.T, origin='lower', extent=extent, cmap="YlOrBr", 
                       vmin=umin, vmax=umax, interpolation='bilinear')
        
        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
            
        # Overlay the imposed data points that are close to this time slice
        if extra.imp_size > 0:
            imp_t = extra.imp_points[:, 0]
            # Select points within the time window
            points_in_slice = extra.imp_points[np.abs(imp_t - t_val) <= time_window]
            if len(points_in_slice) > 0:
                # Plot x (col 1) vs y (col 2)
                ax.scatter(points_in_slice[:, 1], points_in_slice[:, 2], 
                           s=1.5, alpha=0.8, edgecolor="none", facecolor="black", zorder=100)

    # Add a single colorbar for the entire figure
    fig.subplots_adjust(right=0.9, top=0.9) # top=0.9 also helps with suptitle
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='u')
    fig.savefig(path0, bbox_inches="tight")
    plt.close(fig)


    # --- Plot 2: Conductivity k(u) ---
    # This part is dimension-agnostic and requires no changes.
    fig, ax = plt.subplots(figsize=(1.7, 1.5))
    ref_uk = extra.ref_uk
    ref_k = get_ref_k(ref_uk)
    if args.infer_k:
        (k,) = domain.neural_net(state, "k_net")(ref_uk)
        k = transform_k(k, mod, args.kmax)
        print(f"DEBUG: Inferred k values (min/max): {np.min(k):.4f} / {np.max(k):.4f}")
    else:
        k = None
    if k is not None:
        ax.plot(ref_uk, k, zorder=10, label='Inferred')
    ax.plot(ref_uk, ref_k, c="C2", lw=1.5, zorder=1, label='Reference')
    ax.set_xlabel("u")
    ax.set_ylabel("k")
    ax.set_ylim(0, 0.03)
    ax.set_title(title1)
    # Optional: add a legend if both are plotted
    # ax.legend() 
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    
    # The rest of the function for saving data/checkpoints can remain the same
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
        d["k"] = k
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
            state_u = domain.field(state, key)
        elif args.solver == "pinn":
            net = state.fields["u_net"]
            arrays = domain.arrays_from_field(net)
            state_u = eval_u_net(domain, net, arrays)
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
    Loads fields from file `path` and interpolates them to the 3D shape of `domain`.
    Works for (t, x, y) domains.

    keys: `list` of `str`
        Keys of fields to load.
    """
    # MODIFIED: Import the N-dimensional regular grid interpolator
    from scipy.interpolate import RegularGridInterpolator

    src_state = odil.State(fields={key: odil.Field() for key in keys})
    state = odil.State(fields={key: odil.Field() for key in keys})
    odil.core.checkpoint_load(domain, src_state, path)

    # Get the 1D coordinate arrays for the TARGET domain
    target_coords = domain.points_1d()

    for key in keys:
        src_u = src_state.fields[key]
        
        # Check if the source data has a valid shape
        if len(src_u.array.shape) != domain.ndim:
            raise ValueError(
                f"Dimension mismatch: Source field '{key}' has {len(src_u.array.shape)} dims, "
                f"but target domain has {domain.ndim} dims."
            )
            
        # Create a temporary domain object representing the SOURCE data's grid
        src_domain = odil.Domain(
            cshape=src_u.array.shape,
            dimnames=domain.dimnames, # Use same dimension names
            lower=domain.lower,
            upper=domain.upper,
            dtype=domain.dtype,
            mod=odil.backend.ModNumpy(),
        )
        src_u = src_domain.init_field(src_u)

        if src_domain.cshape != domain.cshape:
            printlog(f"Interpolating field '{key}' from {src_domain.cshape} to {domain.cshape}")
            
            # Get the 1D coordinate arrays from the SOURCE domain
            src_coords = src_domain.points_1d()

            # Create the 3D interpolator object
            # It takes a tuple of coordinate arrays and the data array
            interpolator = RegularGridInterpolator(
                points=src_coords, 
                values=src_u.array, 
                method='linear', 
                bounds_error=False, 
                fill_value=0
            )

            # Create a meshgrid of points where we want to evaluate the interpolator
            # 'indexing='ij'' is crucial to match the (t, x, y) array ordering
            target_mesh = np.meshgrid(*target_coords, indexing='ij')
            
            # Stack the meshgrid arrays into a list of (t, x, y) points
            target_points = np.stack([grid.ravel() for grid in target_mesh], axis=-1)

            # Evaluate the interpolator on the new grid points
            interp_values = interpolator(target_points)
            
            # Reshape the flattened result back to the target domain's shape
            state.fields[key].array = interp_values.reshape(domain.cshape)
        else:
            # If shapes match, no interpolation is needed
            state.fields[key] = src_u
            
    return state


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx, args.Ny), dimnames=("t", "x", "y"), multigrid=args.multigrid, dtype=dtype)
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)
    mod = domain.mod
    # Evaluate exact solution, boundary and initial conditions.
    tt, xx, yy = domain.points()
    init_u = get_init_u(tt, xx, yy, mod)

    # --- (New Corrected Block) ---
    # Load reference solution.
    if args.ref_path is not None:
        printlog("Loading reference solution from '{}'".format(args.ref_path))
        with open(args.ref_path, 'rb') as f:
            ref_data = pickle.load(f)
        
        if 'state_u' not in ref_data:
            raise KeyError(f"Reference file {args.ref_path} does not contain the key 'state_u'")

        src_array = ref_data['state_u']

        if src_array.shape != domain.cshape:
            printlog(f"Interpolating reference field from {src_array.shape} to {domain.cshape}")
            from scipy.interpolate import RegularGridInterpolator
            
            # Create temporary source domain to get its coordinates
            src_domain = odil.Domain(cshape=src_array.shape, dimnames=domain.dimnames,
                                    lower=domain.lower, upper=domain.upper)
            
            src_coords = src_domain.points_1d()
            target_coords = domain.points_1d()

            interpolator = RegularGridInterpolator(points=src_coords, values=src_array, method='linear',
                                                bounds_error=False, fill_value=0)
            
            target_mesh = np.meshgrid(*target_coords, indexing='ij')
            target_points = np.stack([grid.ravel() for grid in target_mesh], axis=-1)
            interp_values = interpolator(target_points)
            ref_u = interp_values.reshape(domain.cshape)
        else:
            # Shapes match, no interpolation needed
            ref_u = src_array
    else:
            ref_u = get_init_u(tt, xx, yy, mod)

    ref_u = domain.cast(ref_u) # Ensure correct data type

    # Add noise after choosing points with imposed values.
    imp_u = ref_u
    if args.noise:
        rng = np.random.default_rng(args.seed)
        imp_u += rng.normal(loc=0, scale=args.noise, size=ref_u.shape)

    imp_mask, imp_points, imp_indices = get_imposed_mask(args, domain)
    imp_size = len(imp_points)
    with open("imposed.csv", "w") as f:
        f.write(",".join(domain.dimnames) + "\n")
        for p in imp_points:
            f.write("{:},{:},{:}".format(*p) + "\n")

    ref_uk = np.linspace(0, 1, 200).astype(domain.dtype)
    ref_k = get_ref_k(ref_uk)

    extra = argparse.Namespace()

    def add_extra(d, *keys):
        for key in keys:
            setattr(extra, key, d[key])

    add_extra(
        locals(),
        "args",
        "ref_u",
        "ref_uk",
        "ref_k",
        "init_u",
        "imp_mask",
        "imp_size",
        "imp_u",
        "imp_indices",
        "imp_points",
    )
    extra.epoch = mod.variable(domain.cast(0))

    if args.solver == "pinn":
        t_inner, x_inner = domain.random_inner(args.Nci)
        t_bound0, x_bound0 = domain.random_boundary(1, 0, args.Ncb)
        t_bound1, x_bound1 = domain.random_boundary(1, 1, args.Ncb)
        t_bound = np.hstack((t_bound0, t_bound1))
        x_bound = np.hstack((x_bound0, x_bound1))
        t_init, x_init = domain.random_boundary(0, 0, args.Ncb)
        u_init = get_init_u(t_init, x_init, mod)
        u_bound = get_init_u(t_bound, x_bound, mod)
        printlog("Number of collocation points:")
        printlog("inner: {:}".format(len(t_inner)))
        printlog("init: {:}".format(len(t_init)))
        printlog("bound: {:}".format(len(t_bound)))
        add_extra(locals(), "t_inner", "x_inner", "t_bound", "x_bound", "t_init", "x_init", "u_init", "u_bound")

    state = odil.State()
    if args.solver == "odil":
        operator = operator_odil
        state.fields["u"] = np.zeros(domain.cshape)
    elif args.solver == "pinn":
        state.fields["u_net"] = domain.make_neural_net([2] + args.arch_u + [1])
        operator = operator_pinn
    else:
        raise RuntimeError(f"Unknown solver={solver}")

    if args.infer_k:
        state.fields["k_net"] = domain.make_neural_net([1] + args.arch_k + [1])
    if args.infer_s:
        state.fields["s_net"] = domain.make_neural_net([domain.ndim] + args.arch_s + [1])

    state = domain.init_state(state)

    problem = odil.Problem(operator, domain, extra)

    if args.checkpoint is not None:
        printlog("Loading checkpoint '{}'".format(args.checkpoint))
        odil.core.checkpoint_load(domain, state, args.checkpoint)
        tpath = os.path.splitext(args.checkpoint)[0] + "_train.pickle"
        if args.checkpoint_train is None:
            assert os.path.isfile(tpath), "File not found '{}'".format(tpath)
            args.checkpoint_train = tpath

    if args.checkpoint_train is not None:
        printlog("Loading history from '{}'".format(args.checkpoint_train))
        history.load(args.checkpoint_train)
        args.epoch_start = history.get("epoch", [args.epoch_start])[-1]
        frame = history.get("frame", [args.frame_start])[-1]
        printlog("Starting from epoch={:} frame={:}".format(args.epoch_start, args.frame_start))
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