
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

model = tf.keras.Sequential ( [ 
        tf.keras.layers.Input ( shape = ( 2, ) ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 100, activation = "tanh" ),
        tf.keras.layers.Dense ( 2 )
    ] )

def test_error():
    pass

model.load_weights ( "latest.weights.h5" )

print ( "\nTest error:", test_error() )