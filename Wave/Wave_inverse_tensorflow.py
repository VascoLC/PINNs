import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Initial condition
n_ic = 400
x_ic = 2*np.random.rand(n_ic,1)-1
t_ic = np.zeros((n_ic,1), dtype=np.float32)
xt_ic_np = np.hstack([x_ic,t_ic]).astype(np.float32)

def U_IC(x,t):
    u_exact = np.zeros_like(x_ic)
    for k in range(1,6):
         u_exact+=1/10*(np.cos((x-t+0.5)*np.pi*k)+np.cos((x+t+0.5)*np.pi*k))
    return u_exact

xt_ic = tf.convert_to_tensor(xt_ic_np)
u_ic   = tf.convert_to_tensor(U_IC(x_ic,t_ic))

# Boundary Conditions
n_bc = 400
t_bc = np.random.rand(n_bc,1).astype(np.float32)
edges = np.random.choice(["left","right"], size=n_bc)

x_bc = np.random.rand(n_bc,1)
x_bc[edges=="left"]   = -1.0
x_bc[edges=="right"]  = 1.0
xt_bc_np = np.hstack([x_bc,t_bc]).astype(np.float32)

u_bc_np = np.zeros((n_bc,1), dtype=np.float32)
