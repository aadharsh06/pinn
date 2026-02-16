### Discrete Time Models - Introduction

Raissi _et al._ (2019) defines discrete time models.

In discrete time PINNs, time is treated differently. Instead of forcing the differential equation to hold at all points in the domain, the model is trained to respect dictated by a numerical time stepping method. The neural network predicts the solution at a set of discrete time stages, and the physics is enforced by requiring that these predictions satisfy a known update rule (we use Runge-Kutta) that approximates how the system moves from one time level to the next.

### Using Runge-Kutta

We use Implicit Runge-Kutta methods, with a given set of coefficients.

Let us consider Runge-Kutta of $q$ stages. Our single time step ranges from,
$t_n$ to $t_{n+1}$. \
Thus, $\Delta t$ = $t_n - t_{n+1}$ 

We define,
$c_j \in [0,1]$ for $j \le q$; here $c_j$ are the Runge–Kutta stage coefficients that specify the relative time locations of the intermediate stages within a single time step.

We treat $u(t_n + c_j\,\Delta t, x)$ are the intermediate functions, we treat them as independent functions. We will enforce RK contraints and boundary constraints on these to make them follow the time stepping scheme.

Thus, our network outputs  : $\big[\ u^{n+c_1}(x), \ldots, u^{n+c_q}(x), u^{n+1}(x) \,\big].$
(All time steps)

### Boundary conditions

We are given $x_0, u_0$, we have a snapshot of the solution at time $t_n$. (A few hundred points usually)

Let $x_1$ be the set of boundary points on the spatial domain.

### How do we actually enforce the Runge-Kutta constraints?

We have the general RK with $q$ stages equations,
$u^n(x_0) = u^{n+c_i}(x_0) - \Delta t \sum_j a_{ij}\mathcal{N}[u^{n+c_j}(x_0)])$ ---- (1)

For the final step,
$u^{n+1} = u^n + \Delta t \sum_{j=1}^{q} b_j\mathcal{N}[ u^{n+c_j}]$ ---- (2)

Where $a_{ij}$ and $b_j$ are the coefficients/weights, which we will discuss in the coming section.

Consider the column vector $\mathbf{U}$ defined as (we are just stacking the outputs)

$$
\mathbf{U} =
\begin{bmatrix}
u^{n+c_1} \\
u^{n+c_2} \\
\vdots \\
\vdots \\
u^{n+c_q} \\
u^{n+1}
\end{bmatrix}.
$$

Define $\mathbf{F}(\mathbf{U})$ as the natural (PDE) operator applied termwise,

$$
\mathbf{F}(\mathbf{U}) =
\begin{bmatrix}
\mathcal{N}\left(u^{n+c_1}\right) \\
\mathcal{N}\left(u^{n+c_2}\right) \\
\vdots \\
\vdots \\
\mathcal{N}\left(u^{n+c_q}\right) \\
\mathcal{N}\left(u^{n+1}\right)
\end{bmatrix}.
$$

Thus, we can write

$$
\mathbf{U} -
\Delta t
\begin{bmatrix}
\mathbf{A} \\
\mathbf{b}^T
\end{bmatrix}
\mathbf{F}(\mathbf{U}) =
\mathbf{1}_{q+1}u^n
$$

This is just equation (1) and (2) in a matrix representation, so that operations can be vectorized.

Here, $A \in \mathbb{R}^{q \times q}$ contains the Runge–Kutta stage weights, and $\mathbf{b}^T \in \mathbb{R}^{1 \times q}$ contains the final weights.

### About weights/coefficients

From an intuitive understanding, we can say, \
$a_{ij}$ - How much stage $j$ influences stage $i$. \
$b_j$ - How much stage $j$ contributes to the final step.

Since, we are using implicit RK, all these weights are already computed and provided to us. \
They are precomputed using classical numerical analysis.
