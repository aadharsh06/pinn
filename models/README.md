
# Burgers' Equation

## Background

Burgers' Equation is a Partial Differential Equation PDE which occurs in various fields of physics, but most importantly fluid mechanics. We use the general form of Burgers' equation also known as the "Viscous Burgers' equation", 

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}
= \nu \frac{\partial^2 u}{\partial x^2}$$

Raissi _et al._ (2019), in one of the most fundamental papers on PINNs, implements this equation with $\nu = \frac{0.01}{\pi}$. We draw inspiration from many of their ideas in this implementation.

The PINN models the state variable $u$. We define,
$$
f :\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} -  \nu \frac{\partial^2 u}{\partial x^2}
$$
where the Mean Squared Error (MSE) of $f$ will be our collocation points error.

The boundary conditions are Dirichlet,
$$
x \in [-1,1], \quad t \in [0,1]
$$

$$
u(0,x) = -\sin(\pi x)
$$

$$
u(t,-1) = u(t,1) = 0
$$


## Models
We use Tensorflow in Python for implementations.

The test set used: [Link to test set](https://github.com/maziarraissi/PINNs/tree/master/appendix/Data)
We use relative $L_2$ error, the same as used in Raissi _et al._ (2019).

The best model was obtained in **Implementation IV**.
The weights for each implementation are stored in the `weights/` folder. Please use them with the corresponding model architecture.

### Implementation I: Using Adam with a fixed learning rate
The Adam (Adaptive Momentum) optimizer is one of the best general purpose optimizers in deep learning. 

Architecture and hyperparameters used:
*  5 Hidden layers, 20 neurons each, with $tanh$ activation.
* 5000 epochs with Adam optimizer
* 10,000 collocation points with 100 points for each boundary condition (Total 300 initial and boundary points).
* Learning rate fixed = 0.01

**Test set error**: 0.255668 (or $2.5 \times 10^{-1}$)

### Implementation II: Using Adam with a flexible learning rate

Model architecture details are same as in Implementation I.
The only difference is we change the optimizers (Adam) learning rate from 0.01 to 0.001 after 2000 epochs.

Idea: Initially, we take big steps using the high learning rate, and then we fine tune down to the optimum weights using the smaller learning rate.

**Test set error**: 0.088083 (or $8.8 \times 10^{-2}$)

### Implementation III: Using Adam with a flexible learning rate and **L-BFGS**

The Limited - Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) is a quasi-Newton optimization algorithm which generally works well for PINNs, and is the primary optimizer used in Raissi _et al._ (2019).

Adam is used to train for 10,000 epochs with of those 2,000 epochs being run at learning rate = 0.01 and the rest at 0.001. Then, we carry over the trained variables to L-BFGS which trains for 1,000 iterations. 

Note that 1 Adam iteration is not the same as one L-BFGS iteration. Each iteration involves solving a quasi-Newton step, which is more computationally expensive, and converges faster.
        
Tensorflow does not have a native implementation of this optimizer. Instead, we need to import a module called `tensorflow_probability`, which contains it.

L-BFGS minimizes $L(\theta)$, where it expects $\theta$ to be a flat vector, that is $\theta \in \mathbb{R}^n$, where $n \in \mathbb{N}$ as contrasted to our model's trainable variables which is generally a 2D matrix for dense networks. Thus, when shifting from Adam to L-BFGS we need to flatten the model's trainable variables for training and unflatten them after an optimization step is complete, for our model to update them.


Architecture and hyperparameters used:
*  9 Hidden layers, 20 neurons each, with $tanh$ activation.
* 10000 epochs with Adam optimizer, 1000 with L-BGFS
* 10,000 collocation points with 100 points for each boundary condition (Total 300 initial and boundary points).
* Adam learning rate policy, same as Implementation II.

**Test set error**: 0.013301 (or $1.3 \times 10^{-2}$)

### Implementation IV: Using only L-BFGS

Model architecture details are same as in Implementation III.
The only difference is we change the optimizers (L-BFGS) max iterations from 1000 to 5000.

**Test set error**: 0.005169 (or $5.1 \times 10^{-3}$).

## Plots
Under `plots/` there are some training error plots for different implementations over the number of epochs.
`plots_soln.py` Generates a 2D shaded plot for x, t and the solution. Can be modified to show actual and predicted solution plots. Implementation IV's plot, generated from this, has been shown in `plots/`.