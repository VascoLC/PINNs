# plot_fd_fullfield.py
import numpy as np
import matplotlib.pyplot as plt
from FiniteDifferences import conservative_CN

# Run solver (adjust Nx, Nt for resolution)
x, t, U = conservative_CN(Nx=201, Nt=200, T=1.0)

# === Plot full space–time temperature field ===
X, T = np.meshgrid(x, t)   # X: (Nt+1,Nx), T: (Nt+1,Nx)

plt.figure(figsize=(8,6))
# pcolormesh shows every grid point; shading='auto' avoids gaps
pcm = plt.pcolormesh(X, T, U, cmap="inferno", shading="auto")
plt.colorbar(pcm, label="Temperature u(x,t)")
plt.xlabel("x")
plt.ylabel("time t")
plt.title("Full temperature field u(x,t) from FD solver")
plt.tight_layout()
plt.show()
