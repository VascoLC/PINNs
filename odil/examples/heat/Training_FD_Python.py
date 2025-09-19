import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1  # Length of rod
T = 1  # Maximum time bound
m = 5  # No. of intervals in space
n = 5  # No. of intervals in time
h = L / m  # Step size in space
k = T / n  # Step size in time
b = 0.05  # Coefficient of diffusivity
mu = k / h**2  # k = k / h * h

# Check stability
c = b * mu
if c <= 0 or c >= 0.5:
    print('Scheme is unstable')

# Initialization of solution
v = np.zeros((m + 1, n + 1))

# Initial condition
ic1 = lambda x: np.sin(np.pi * x)

# Space discretization
for j in range(1, m + 2):
    v[0, j - 1] = ic1((j - 1) * h)

# Boundary condition
b1 = lambda t: 0  # L.B.C
b2 = lambda t: 0  # R.B.C

# Time discretization
for i in range(1, n + 2):
    v[i - 1, 0] = b1((i - 1) * k)
    v[i - 1, n] = b2((i - 1) * k)

# Implementation of the scheme
for i in range(n):
    for j in range(1, m):
        v[i + 1, j] = (1 - 2 * b * mu) * v[i, j] + b * mu * v[i, j + 1] + b * mu * v[i, j - 1]

# Visualization
x = np.linspace(0, L, m + 1)
t = np.linspace(0, T, n + 1)
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, v, cmap='viridis')
ax.set_xlabel('Space X')
ax.set_ylabel('Time T')
ax.set_zlabel('V')
plt.title('Python Program for Heat Equation')
plt.show()