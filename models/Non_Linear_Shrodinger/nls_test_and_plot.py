
# Calculating test error and generating plots

# Python 3.10 or below

import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model

model = tf.keras.Sequential ( [ 
        tf.keras.layers.Input ( shape = ( 2, ) ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 2 )
    ] )

# Loading weights

model.load_weights ( "latest.weights.h5" )

# Loading test set

data = loadmat ( "NLS.mat" )
t = data['tt']
x = data['x']
uu = data['uu']

uu = np.abs ( uu )

# L2 relative error calculation

X, T = np.meshgrid ( x.squeeze(), t.squeeze(), indexing = 'ij' )
XT = np.hstack ( [ X.reshape ( -1, 1 ), T.reshape ( -1, 1 ) ] )

pred = model.predict ( XT )
u_pred = pred[:, 0:1]
v_pred = pred[:, 1:2]

pred = np.sqrt ( np.square ( u_pred ) + np.square ( v_pred ) ).reshape ( uu.shape )

error = np.linalg.norm ( pred - uu ) / np.linalg.norm ( uu )

print ( "\nTest set error: ", error, "\n" )

# Plotting shade plots actual vs predicted

x_plot = x.squeeze()
t_plot = t.squeeze()

fig, axs = plt.subplots ( 1, 2, figsize = ( 16, 6 ) )

im0 = axs[0].pcolormesh ( t_plot, x_plot, uu )
axs[0].set_title ( "Ground Truth |u(x,t)|" )
axs[0].set_xlabel ( "t" )
axs[0].set_ylabel ( "x" )
fig.colorbar ( im0 )

im1 = axs[1].pcolormesh ( t_plot, x_plot, pred )
axs[1].set_title ( "Predicted |u(x,t)|" )
axs[1].set_xlabel ( "t" )
axs[1].set_ylabel ( "x" )
fig.colorbar ( im1 )

plt.show()
plt.savefig ( "./plots/nls_pred_vs_truth.png" )

plt.clf()

# Plotting a slice, at t = 0.79, (value chosen from Rassi et al (2019) )

t_val = 0.79

# Choosing a value very close to 0.79 from our plot set
t_val = np.argmin ( np.abs ( t_plot - t_val ) )

uu_slice = uu[:, t_val]
pred_slice = pred[:, t_val]
plt.figure ( figsize = ( 5, 5 ) )

plt.plot ( x_plot, uu_slice, 'k-', label = "Ground Truth |u|" )
plt.plot ( x_plot, pred_slice, 'r--', label = "Predicted |u|" )
plt.xlabel ( "x" )
plt.ylabel ( "|u(x,t)|" )

plt.title ( "Slice at t = 0.79" )
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
plt.savefig ( "./plots/nls_slice_t_0.79.png" )