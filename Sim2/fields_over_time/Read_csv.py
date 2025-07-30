import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of files and times
files = [
    ("field_at_t_0.0.csv", 0.0),
    ("field_at_t_0.2.csv", 0.2),
    ("field_at_t_0.4.csv", 0.4),
    ("field_at_t_0.6.csv", 0.6),
    ("field_at_t_0.8.csv", 0.8),
    ("field_at_t_1.0.csv", 1.0),
]

nx = ny = 100  # You can adjust if you know your grid size

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (fname, t) in zip(axes, files):
    # Read the data
    df = pd.read_csv(fname)
    # Assuming the file has columns: x, y, t, u, k
    # Reshape u to (nx, ny)
    u = df['u'].values.reshape(nx, ny)
    x = df['x'].values.reshape(nx, ny)
    y = df['y'].values.reshape(nx, ny)

    im = ax.contourf(x, y, u, levels=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"t = {t}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
