
# Burger equation PINN: training and test set error.

# Python 3.10 or less required.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow_probability as tfp

# Set epochs for adam and lbfgs

adam_epochs = 5000
lbfgs_epochs = 0

# Nf - Number of collocation points over the space time domain
# Nu_ind - Number of boundary points for each boundary / initial condition

Nf = 10000
Nu_ind = 100

# Kinematic viscosity
c = tf.constant ( 0.01 / np.pi, dtype = tf.float32 )

# Defining all initial and boundary conditions

x = np.random.uniform ( -1, 1, ( Nu_ind, 1 ) )
t = np.zeros ( ( Nu_ind, 1 ) )

ux_train = np.hstack ( ( x, t ) )
uy_train = - np.sin ( np.pi * x )

x = np.full ( shape = ( Nu_ind, 1 ), fill_value = -1 )
t = np.random.uniform ( 0, 1, ( Nu_ind, 1 ) )

ux_train = np.vstack ( ( ux_train, np.hstack ( ( x, t ) ) ) )

x = np.full ( shape = ( Nu_ind, 1 ), fill_value = 1 )
t = np.random.uniform ( 0, 1, ( Nu_ind, 1 ) )

ux_train = np.vstack ( ( ux_train, np.hstack ( ( x, t ) ) ) )
uy_train = np.vstack ( ( uy_train, np.zeros ( ( Nu_ind * 2, 1 ) ) ) )

# Collocation points

x = np.random.uniform ( -1, 1, ( Nf, 1 ) )
t = np.random.uniform ( 0, 1, ( Nf, 1 ) )

fx_train = np.hstack ( ( x, t ) )

# Model architeture 

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

# Functions for lbfgs 

# Flatten our model weights into a single vector

def pack_weights ( vars ):
    return tf.concat ( [ tf.reshape ( v, [-1] ) for v in vars ], axis = 0 )

# Unflatten weights

def unpack_weights ( vars, flat ):
    idx = 0
    for v in vars:
        size = tf.size ( v )
        shape = v.shape
        v.assign ( tf.reshape ( flat[idx:idx+size], shape ) )
        idx += size

# passing gradients for lbfgs 

def value_and_gradients ( flat_weights ):
    
    unpack_weights ( model.trainable_variables, flat_weights )

    with tf.GradientTape() as tape:
        loss = MSE_u() + MSE_f()

    grads = tape.gradient ( loss, model.trainable_variables )
    return loss, pack_weights ( grads )

# Mean squared error for boundary / initial conditions

def MSE_u():
    return tf.reduce_mean ( tf.square ( model ( ux_train ) - uy_train ) )

# Mean squared error for collocation points

def MSE_f():
    x_tf = tf.convert_to_tensor ( x, dtype = tf.float32 )
    t_tf = tf.convert_to_tensor ( t, dtype = tf.float32 )
    
    with tf.GradientTape ( persistent = True ) as tape2:
        tape2.watch ( [ x_tf, t_tf ] )
        with tf.GradientTape ( persistent = True ) as tape1:
            tape1.watch ( [ x_tf, t_tf ] )
            u_pred = model ( tf.concat ( [x_tf, t_tf], axis = 1 ) )
        
        du_dx = tape1.gradient ( u_pred, x_tf )
        du_dt = tape1.gradient ( u_pred, t_tf )
    
    d2u_dx2 = tape2.gradient ( du_dx, x_tf )
    return tf.reduce_mean ( tf.square ( du_dt + u_pred * du_dx - c * d2u_dx2 ) )

# Loading test set, computing L2 error

def test_error():
    
    data = loadmat ( "burgers_shock.mat" )
    x_val = data['x']
    t_val = data['t']
    u_exact = data['usol']
    
    X, T = np.meshgrid ( x_val.squeeze(), t_val.squeeze(), indexing = 'ij' )
    XT = np.hstack ( [ X.reshape ( -1, 1 ), T.reshape ( -1, 1 ) ] )

    u_pred = model.predict ( XT ).reshape ( u_exact.shape )
    return np.linalg.norm ( u_pred - u_exact ) / np.linalg.norm ( u_exact ) 

# Adam optimizer

opt = tf.keras.optimizers.Adam ( learning_rate = 0.01 )

print ( "INFO: Training Started" )

# For plotting

epoch_list = []
total_list = []

# Training with adam

for epoch in range ( 1, adam_epochs + 1 ):
    with tf.GradientTape() as tape:
        loss = MSE_u() + MSE_f()
    grads = tape.gradient ( loss, model.trainable_variables )
    opt.apply_gradients ( zip ( grads, model.trainable_variables ) )
    
    if ( epoch % 100 == 0 ):
        print ( "INFO: Adam Epoch {} reached, MSE_u : {}, MSE_f: {}".format ( str ( epoch ), str ( float ( MSE_u() ) ), str ( float ( MSE_f() ) ) ) )
        epoch_list.append ( epoch )
        total_list.append ( float ( MSE_u() ) + float ( MSE_f() ) )
        
    # Switch to 0.001 learning rate after 2000 epochs
    if ( epoch == 2000 ):
        print ( "INFO: Adam Learning rate now 0.001" )
        opt.learning_rate.assign ( 0.001 )

print ( "INFO: Training with Adam done" )

# Train with lbfgs (If using only Adam, recommended to comment out from here till line 180)

lbfgs_train = tfp.optimizer.lbfgs_minimize (
    value_and_gradients,
    initial_position = pack_weights ( model.trainable_variables ),
    max_iterations = lbfgs_epochs,
    tolerance = 1e-8
)

# Save weights

print ( "INFO: Overall Training Done" )

unpack_weights ( model.trainable_variables, lbfgs_train.position )

model.save_weights ( "burger_latest.weights.h5" )
print ( "\nTest error:", test_error() )

# Final plots

plt.plot ( epoch_list, total_list, label = "Training loss" )
plt.xlabel ( "Epochs" )
plt.ylabel ( "Loss" )
plt.legend()
plt.show()