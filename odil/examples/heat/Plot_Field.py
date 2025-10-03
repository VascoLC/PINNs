#Plot_Field

import pickle
import matplotlib.pyplot as plt
import numpy as np

data = 'C:\PINNs_Git\PINNs\odil\examples\heat\out_heat_inverse_1D_adam_ODIL/data_00030.pickle'

with open(data, 'rb') as file:
    # Load the data from the file
    my_data = pickle.load(file)

# --- Add these lines ---
print("Type of the loaded object:", type(my_data))
print("\nContents of the pickle file:")
print(my_data)

# If the object is a dictionary, list its keys
if isinstance(my_data, dict):
    print("\nVariables stored in the file (keys):")
    print(list(my_data.keys()))
# ----------------------

print(my_data['state_u'].shape)
