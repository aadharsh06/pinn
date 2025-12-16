
# Program to plot the final predictions

# Python 3.10 or lower

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt

# User correct architecture for corresponding implementation!

model = tf.keras.Sequential ( [
        tf.keras.layers.InputLayer ( input_shape = ( 2, ) ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 20, activation = 'tanh' ),
        tf.keras.layers.Dense ( 1 )
    ] )

model.load_weights ( "weights\\burger_latest.weights.h5" )

data = loadmat ( "burgers_shock.mat" )
x = data['x']
t = data['t']
u = data['usol']

X, T = np.meshgrid ( x.squeeze(), t.squeeze(), indexing = 'ij' )

XT = np.hstack ( [ X.reshape ( -1, 1 ), T.reshape ( -1, 1 ) ] )

# Comment out this line if you want to plot the exact solution
u = model.predict ( XT ).reshape ( u.shape )

plt.pcolormesh ( X, T, u )
plt.colorbar ( label = "u(x,t)" )
plt.xlabel ( "x" )
plt.ylabel ( "t" ) 
plt.title ( "Predicted Burgers solution" )
plt.show()