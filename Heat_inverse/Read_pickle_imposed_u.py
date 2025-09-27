import pandas as pd
import numpy as np
import tensorflow as tf

# --- Configuration ---
pickle_file_path = "C:/PINNs_Git/PINNs/odil/examples/heat/out_heat_direct_1D_256/data_00010.pickle"

# --- Grid Assumptions (MUST MATCH THE GRID THAT imp_u_grid IS BUILT ON) ---
# Assuming imp_u_grid (256, 256) covers x from 0 to 1 and t from 0 to 1
# Adjust these min/max values if your actual grid ranges are different.
x_grid_min, x_grid_max = 0, 1
t_grid_min, t_grid_max = 0, 1

# --- Load and process 'imp_indices' and 'imp_u' ---
try:
    full_data_dict = pd.read_pickle(pickle_file_path)

    print("Successfully loaded pickle file.")
    print("Keys available in full_data_dict:", list(full_data_dict.keys()))

    # Check for the existence of required keys
    if 'imp_indices' not in full_data_dict:
        raise KeyError("'imp_indices' key not found in the pickle file.")
    if 'imp_u' not in full_data_dict:
        raise KeyError("'imp_u' key not found in the pickle file.")

    # Extract imp_u grid data
    imp_u_grid = full_data_dict['imp_u'].astype(np.float32) # Shape: (Ny, Nx) e.g., (256, 256)

    # Extract imp_indices (linear indices)
    imp_indices = full_data_dict['imp_indices'].astype(int) # Shape: (500,)

    # --- Derived Grid Properties ---
    Ny, Nx = imp_u_grid.shape # Number of points in T and X dimensions of the grid

    # Create the 1D coordinate arrays for the grid
    x_grid_coords = np.linspace(x_grid_min, x_grid_max, Nx)
    t_grid_coords = np.linspace(t_grid_min, t_grid_max, Ny)

    # --- Convert linear indices to (row_index, col_index) for the grid ---
    # linear_index = row_index * Nx + col_index
    # row_index = linear_index // Nx
    # col_index = linear_index % Nx

    # Calculate row and column indices for each important point
    imp_row_indices = imp_indices // Nx
    imp_col_indices = imp_indices % Nx

    # --- Extract u values at these indices ---
    u_obs_np = imp_u_grid[imp_row_indices, imp_col_indices].reshape(-1, 1).astype(np.float32) # (500, 1)

    # --- Get corresponding x and t coordinates ---
    # Map row_indices to t-coordinates and col_indices to x-coordinates
    x_obs_np = x_grid_coords[imp_col_indices].reshape(-1, 1).astype(np.float32)
    t_obs_np = t_grid_coords[imp_row_indices].reshape(-1, 1).astype(np.float32)

    # Combine x and t into XT_obs_np
    XT_obs_np = np.hstack([x_obs_np, t_obs_np]).astype(np.float32) # (500, 2)

    print(f"\nExtracted XT_obs_np shape: {XT_obs_np.shape}")
    print(f"Extracted u_obs_np shape: {u_obs_np.shape}")
    print("First 5 rows of XT_obs_np (x, t values from imp_indices):")
    print(XT_obs_np[:5])
    print("First 5 rows of u_obs_np (u values at imp_indices):")
    print(u_obs_np[:5])

    # Convert to TensorFlow tensors
    XT_obs = tf.convert_to_tensor(XT_obs_np)
    u_obs  = tf.convert_to_tensor(u_obs_np)

    # --- Handling k_obs (if it corresponds to these imp_indices) ---
    # If there's a 'ref_k' or 'k' grid that also corresponds to imp_u_grid's dimensions
    # and you want to use it for supervision at the imposed points.
    if 'ref_k' in full_data_dict: # Or 'k' if that's the relevant key for k-grid
        k_grid = full_data_dict['ref_k'].astype(np.float32)
        if k_grid.shape == imp_u_grid.shape:
            k_obs_np = k_grid[imp_row_indices, imp_col_indices].reshape(-1, 1).astype(np.float32)
            k_obs = tf.convert_to_tensor(k_obs_np)
            print(f"Extracted k_obs_np shape: {k_obs_np.shape}")
            print("First 5 rows of k_obs_np (k values at imp_indices):")
            print(k_obs_np[:5])
        else:
            print(f"Warning: 'ref_k' grid shape {k_grid.shape} does not match 'imp_u' grid shape {imp_u_grid.shape}. Setting k_obs to None.")
            k_obs = None
    elif 'k' in full_data_dict: # Check for 'k' if 'ref_k' isn't the one
         k_grid = full_data_dict['k'].astype(np.float32)
         if k_grid.shape == imp_u_grid.shape:
            k_obs_np = k_grid[imp_row_indices, imp_col_indices].reshape(-1, 1).astype(np.float32)
            k_obs = tf.convert_to_tensor(k_obs_np)
            print(f"Extracted k_obs_np shape: {k_obs_np.shape}")
            print("First 5 rows of k_obs_np (k values at imp_indices):")
            print(k_obs_np[:5])
         else:
            print(f"Warning: 'k' grid shape {k_grid.shape} does not match 'imp_u' grid shape {imp_u_grid.shape}. Setting k_obs to None.")
            k_obs = None
    else:
        print("Warning: Neither 'ref_k' nor 'k' grid data found for supervision. Setting k_obs to None.")
        k_obs = None

except FileNotFoundError:
    print(f"Error: The file '{pickle_file_path}' was not found. Please check the path carefully.")
except KeyError as e:
    print(f"Error: Required key missing from pickle file: {e}")
except ValueError as e:
    print(f"Error processing data from pickle file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# --- You would then continue with your PINN training logic ---
# Your compute_loss function would now use these XT_obs, u_obs, and potentially k_obs.