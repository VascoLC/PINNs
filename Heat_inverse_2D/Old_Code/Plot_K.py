import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras


'''
# Conductivity vs Temperature
#df = pd.read_csv("2D_second_step/fields_over_time/field_at_t_0.0.csv")
df = pd.read_csv("2D_first_step/fields_over_time/field_at_t_0.4.csv")
df_t1 = df[np.isclose(df["t"], 0)]

u_field = df[["u"]].values.astype(np.float32)
k_field = df[["k"]].values.astype(np.float32)

def k_np(u):
    return 0.02 * np.exp(-20 * (u - 0.5)**2)

u_test = np.linspace(0, 1, 500) 
k_analytic = k_np(u_test)

plt.figure(figsize=(5,4))
#plt.plot(u_test, k_analytic, 'r-', lw=2, label="Analytical $k(u)$")
plt.scatter(u_field, k_field, s=10, alpha=0.6)
plt.xlabel("Temperature")
plt.ylabel("Condutivity")
plt.title(f"K vs U")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

'''
# Training Loss
df = pd.read_csv("2D_second_step/training_loss_log.csv")
epochs =  df[["epoch"]].values.astype(np.float32)
total_loss = df[["total_loss"]].values.astype(np.float32)

plt.figure(figsize=(5,4))
plt.plot(epochs, total_loss,linewidth=0.5)
plt.xlabel("epochs"); plt.ylabel("error")
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()

plt.show()
'''

def load_and_process_data(filepath):
    """
    Loads a single CSV file, checks for errors, and pivots the data into a grid format.

    Args:
        filepath (str): The full path to the CSV file.

    Returns:
        A dictionary containing the processed grid data (X, Y, u) and local min/max,
        or None if the file cannot be processed.
    """
    # First, check if the file even exists.
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    try:
        sim_data = pd.read_csv(filepath)
        # Check if the file is empty after loading.
        if sim_data.empty:
            print(f"Warning: File is empty: {filepath}")
            return None
        
        # Pivot the data to create a 2D grid for plotting.
        grid = sim_data.pivot(index="y", columns="x", values="u")
        X, Y = np.meshgrid(grid.columns.values, grid.index.values)
        u_field = grid.values
        
        # Return all the necessary information in a structured way.
        return {
            "X": X,
            "Y": Y,
            "u": u_field,
            "vmin": u_field.min(),
            "vmax": u_field.max()
        }
    except Exception as e:
        # Catch any other errors during file processing.
        print(f"Error processing file {filepath}: {e}")
        return None

def plot_all_fields(timestamps, base_path):
    """
    Loads all data for the given timestamps, determines a common color scale,
    and creates a 2x3 grid of plots with a single colorbar.

    Args:
        timestamps (list): A list of time values to plot (e.g., [0.0, 0.2, ...]).
        base_path (str): The path to the directory containing the CSV files.
    """
    # 1. --- DATA LOADING AND PREPARATION ---
    # First, load all data into a list. This way, we only read each file once.
    all_data = []
    for t in timestamps:
        filepath = os.path.join(base_path, f"field_at_t_{t:.1f}.csv")
        processed_data = load_and_process_data(filepath)
        all_data.append(processed_data)

    # Find the global min/max for the colorbar, ignoring any files that failed to load.
    valid_data = [d for d in all_data if d is not None]
    if not valid_data:
        print("Error: No valid data could be loaded. Cannot generate plot.")
        return

    # Set the global color range based on all valid data.
    global_vmin = 0.0  # As per your original code for temperature
    global_vmax = max(d['vmax'] for d in valid_data)
    print(f"Global color range set to: vmin={global_vmin}, vmax={1}")

    # 2. --- PLOTTING ---
    # Create the figure and subplots with a robust layout manager.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fig.suptitle('Heat Distribution Over Time', fontsize=16)

    mappable = None  # To store the artist for the colorbar

    # Loop through the axes, the loaded data, and the timestamps together.
    for ax, data, t in zip(axes.ravel(), all_data, timestamps):
        if data:
            # If data was loaded successfully, create the contour plot.
            mappable = ax.contourf(data['X'], data['Y'], data['u'], levels=100,
                                   vmin=global_vmin, vmax=global_vmax, cmap="YlOrBr")
            ax.set_title(f"t = {t:.1f}")
        else:
            # If data is None (file was missing/empty), show a blank subplot.
            ax.set_title(f"t = {t:.1f} (No Data)")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")

    # 3. --- FINAL COLORBAR ---
    # After the loop, create one colorbar for the entire figure.
    if mappable:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.75)
        cbar.ax.set_ylabel('Temperature u(x,y,t)', fontsize=12)
    
    plt.show()


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define the parameters for your plot here.
    timestamps_to_plot = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    path_to_csvs = "C:/PINNs_Git/PINNs/Heat_inverse_2D/2D_first_step/fields_over_time"
    
    # Run the main plotting function.
    plot_all_fields(timestamps_to_plot, path_to_csvs)


