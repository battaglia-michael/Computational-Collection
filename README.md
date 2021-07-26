# Computational-Collection
Collection of small and useful computational codes for physics or data analysis.

### Gauss-Seidel.py
A function that solves a partial differential equation using the Gauss-Seidel relaxation method with replacement and overrelaxation to speed up convergence. It also outputs an animated contour plot (greyscale) with *matplotlib*.

Included is an example implementation for a steady-state heat equation (Laplace equation).

### SolveLinear.py
A linear equation solver that uses Gaussian elimination or LU decomposition and accepts complex (and float) arrays. It does not change the input arrays so that they can be used subsequently if desired.
There is also the function partial pivot which prevents issues with systems that might lead to dividing by zero.

If run as main it will test and compare the runtime vs matrix size for Gaussian elimination, Gaussian elimination with partial pivot, and LU decomposition with partial pivot on a sample array.
