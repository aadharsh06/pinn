
# PINN train file for the Non Linear Schrodinger Equation (NLS) 

# Python 3.10 or below

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from time import time

# Set epochs for adam and lbfgs

adam_epochs = 0
lbfgs_epochs = 50000

# No - Number of initial points for enforcing the initial condition h(0, x) = 2sech(x)
# Nb_ind = Number of boundary points to enforce the 2 periodic boundary conditions
# Nf - Number of collocation points over the space time domain to enforce the PDE

No = 20000
Nb_ind = 20000
Nf = 20000

# To enfoce initial condition

x = np.random.uniform ( -5, 5, ( No, 1 ) )
t = np.zeros ( ( No, 1 ) )

ox_train = np.hstack ( [ x, t ] )
oy_train = np.hstack ( [ 2 / np.cosh ( x ), np.zeros_like ( x ) ] )

x_1 = np.full ( shape = ( Nb_ind, 1 ), fill_value = -5 )
t_1 = np.random.uniform ( 0, np.pi / 2, ( Nb_ind, 1 ) )

# To enfoce boundary conditions

b1_train = np.hstack ( [ x_1, t_1 ] )

x_2 = np.full ( shape = ( Nb_ind, 1 ), fill_value = 5 )
t_2 = np.random.uniform ( 0, np.pi / 2, ( Nb_ind, 1 ) )

b2_train = np.hstack ( [ x_2, t_2 ] )

# Collocation points

x = np.random.uniform ( -5, 5, ( Nf, 1 ) )
t = np.random.uniform ( 0, np.pi / 2, ( Nf, 1 ) )

fx_train = np.hstack ( [ x, t ] )

# Defining the model architeture

model = tf.keras.Sequential ( [ 
        tf.keras.layers.Input ( shape = ( 2, ) ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 2 )
    ] )

# Initial condition loss

def MSE_o():
    return tf.reduce_mean ( tf.square ( model ( ox_train ) - oy_train ) )

b1 = tf.convert_to_tensor ( b1_train, dtype = tf.float32 ) 
b2 = tf.convert_to_tensor ( b2_train, dtype = tf.float32 )

# Boundary condition loss

def MSE_b():
    with tf.GradientTape ( persistent = True ) as tape:
        tape.watch ( b1 )
        tape.watch ( b2 )
        
        h1 = model ( b1 )
        h2 = model ( b2 )
        
        u1 = h1[:, 0:1]
        v1 = h1[:, 1:2]
        u2 = h2[:, 0:1]
        v2 = h2[:, 1:2]
    
    u1x = tape.gradient(u1, b1)[:, 0:1]
    v1x = tape.gradient(v1, b1)[:, 0:1]
    u2x = tape.gradient(u2, b2)[:, 0:1]
    v2x = tape.gradient(v2, b2)[:, 0:1]
    
    return tf.reduce_mean ( tf.square ( h1 - h2 ) ) + tf.reduce_mean ( tf.square ( u1x - u2x ) + tf.square ( v1x - v2x ) )

x_tf = tf.convert_to_tensor ( x, dtype = tf.float32 )
t_tf = tf.convert_to_tensor ( t, dtype = tf.float32 )

# Collocation point loss / PDE loss

def MSE_f():
    
    with tf.GradientTape ( persistent = True ) as tape1:
        tape1.watch ( [ x_tf, t_tf ] )
        with tf.GradientTape ( persistent = True ) as tape2:
            tape2.watch ( [ x_tf, t_tf ] )
            
            h_pred = model ( tf.concat ( [ x_tf, t_tf ], axis = 1 ) )
            u_pred = h_pred[:, 0:1]
            v_pred = h_pred[:, 1:2]
        
        du_dx = tape2.gradient ( u_pred, x_tf )
        du_dt = tape2.gradient ( u_pred, t_tf )
        dv_dx = tape2.gradient ( v_pred, x_tf )
        dv_dt = tape2.gradient ( v_pred, t_tf )
    
    d2u_dx2 = tape1.gradient ( du_dx, x_tf )
    d2v_dx2 = tape1.gradient ( dv_dx, x_tf )

    del tape1
    del tape2
    
    real_term = ( -dv_dt ) + ( 0.5 * d2u_dx2 ) + ( u_pred ** 3 ) + ( u_pred * v_pred * v_pred )
    img_term = du_dt + ( 0.5 * d2v_dx2 ) + ( v_pred ** 3 ) + ( v_pred * u_pred * u_pred )
    
    return tf.reduce_mean ( tf.square ( real_term ) + tf.square (  img_term ) )
        
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

cur_epochs = 1
start = time()

def value_and_gradients ( flat_weights ):
    
    unpack_weights ( model.trainable_variables, flat_weights )

    with tf.GradientTape() as tape:
        loss_o = MSE_o()
        loss_b = MSE_b()
        loss_f = MSE_f()
        loss = loss_o + loss_b + loss_f 
    
    # Printing status and saving weights as a checkpoint, for every 1000 function calls
    # Note 1 iteration != 1 function call!! Generally 1 iteration has many function calls
    global cur_epochs
    if ( cur_epochs % 1000 == 0 ):
        print ( "INFO: Call {} reached; Time taken: {}\nMSE_o: {}, MSE_b: {}, MSE_f: {}\n".format ( str ( cur_epochs ), str ( time() - start ),  str ( float ( loss_o.numpy() ) ), str ( float ( loss_b.numpy() ) ), str ( float ( loss_f.numpy() ) ) ) )
        model.save_weights ( "latest.weights.h5" )
    cur_epochs += 1
        
    grads = tape.gradient ( loss, model.trainable_variables )

    del tape
    return loss, pack_weights ( grads )

print ( "INFO: Training Started" )

# Adam optimizer
opt = tf.keras.optimizers.Adam ( learning_rate = 0.01 )

# Training with Adam

for epoch in range ( 1, adam_epochs + 1 ):
    with tf.GradientTape() as tape:
        loss_o = MSE_o()
        loss_b = MSE_b()
        loss_f = MSE_f()
        loss = loss_o + loss_b + loss_f 
    grads = tape.gradient ( loss, model.trainable_variables )
    opt.apply_gradients ( zip ( grads, model.trainable_variables ) )
    
    # Every 100 epochs, print status and checkpoint weights
    if ( epoch % 100 == 0 ):
        print ( "INFO: Adam epoch {} reached; Time taken: {}\nMSE_o: {}, MSE_b: {}, MSE_f: {}\n".format ( str ( epoch ), str ( time() - start ),  str ( float ( loss_o.numpy() ) ), str ( float ( loss_b.numpy() ) ), str ( float ( loss_f.numpy() ) ) ) )
        model.save_weights ( "latest.weights.h5" )
        
    # Switch to 0.001 learning rate after 2000 epochs
    if ( epoch == 2000 ):
        print ( "INFO: Adam Learning rate now 0.001" )
        opt.learning_rate.assign ( 0.001 )

print ( "INFO: Training with Adam done" )

# Training with lbfgs

lbfgs_train = tfp.optimizer.lbfgs_minimize (
    value_and_gradients,
    initial_position = pack_weights ( model.trainable_variables ),
    max_iterations = lbfgs_epochs,
)

print ( "INFO: Training Done" )

unpack_weights ( model.trainable_variables, lbfgs_train.position )

# Save weights

model.save_weights ( "latest.weights.h5" )