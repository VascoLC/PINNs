import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load the CSV file ---
df = pd.read_csv("C:\PINNs_Git\PINNs\Poisson\Poisson_variants/predicted_field.csv")

# --- Extract columns ---
x = df["x"].values
y = df["y"].values
U = df["u_pred"].values

# --- Create a regular grid for plotting ---
# (Assuming x and t form a structured grid)
x_unique = np.sort(df["x"].unique())
y_unique = np.sort(df["y"].unique())

X, Y = np.meshgrid(x_unique, y_unique)
U_grid = U.reshape(len(y_unique), len(x_unique))

# --- Plot the field ---
plt.figure(figsize=(8, 5))
plt.pcolormesh(X, Y, U_grid, shading="auto", cmap="viridis")
plt.colorbar(label="Predicted Temperature (U_Pred)")
plt.xlabel("x (space)")
plt.ylabel("t (time)")
plt.title("Predicted Temperature Field U(x, t)")
plt.tight_layout()
plt.show()
