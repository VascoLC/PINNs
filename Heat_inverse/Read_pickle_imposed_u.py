import pickle
import matplotlib.pyplot as plt
import numpy as np

data = 'C:\PINNs_Git\PINNs\odil\examples\heat\out_heat_inverse_1D_adam_MODIL/data_00030.pickle'

with open(data, 'rb') as file:
    # Load the data from the file
    my_data = pickle.load(file)

k= my_data['k']

u = my_data['state_u']
u_test = np.linspace(0,1,200)

plt.figure(figsize=(5,4))
plt.plot(u_test, k, 'r-', lw=2, label="Analytical $k(u)$")
plt.xlabel("Temperature")
plt.ylabel("Condutivity")
plt.title("Analytical vs. Obtained Conductivity")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()