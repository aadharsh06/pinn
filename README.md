
# Physics Informed Neural Networks (PINNs)

This repository contains all codes, models, derivations, etc., relating to my work with Physics-Informed Neural Networks. The methodology used and documentation of my work pertaining to each model have been provided in detail in respective README files.

## General Overview of PINNs...what are they exactly?

In general, traditional neural networks can be viewed as function approximators. Given sufficient data, a neural network attempts to approximate an underlying function, even when this function is not known explicitly. \
\
Physics-Informed Neural Networks (PINNs) aim to reduce the dependence on large amounts of data when modeling physical systems by leveraging known physical equations. This is achieved by embedding the partial differential equations (PDEs), corresponding to the given physical system, and their associated boundary and initial conditions directly into the loss function. These physics-based constraints introduce an additional loss term, reducing reliance on real-world data while enforcing physical consistency. \
\
PINNs often produce low test set error with less data with physically consistent solutions.

## Models

PINNs can be classified into continuous time models and discrete time models, based on how we handle the time domain.
In continous time models, the PDE is enforced pointwise in continuous time.
In discrete time models, PDE is enforced through a time stepping scheme (we use Runge Kutta here), which constrains the neural network outputs at discrete time levels.

Implemented models can be found in the respective folders. Please do read the README file in each model's folder :) \
I have attached relevant plots/derivations, wherever applicable.

#### Implemented:
1. Burgers' PDE (continuous time model)
2. Burgers' PDE (discrete time model)
3. Nonlinear Schrödinger equation (continuous time model)
4. Allen–Cahn equation (discrete time model) — In progress.

