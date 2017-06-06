---
title: Non-stationary Kernels
---

## Polynomial Kernel

A very popular non-stationary kernel used in machine learning, the polynomial represents the data features as polynomial expansions up to an index $d$.

$$
	C(\mathbf{x},\mathbf{y}) = (\mathbf{x}.\mathbf{y} + a)^{d}
$$

```scala
val fbm = new PolynomialKernel(2, 0.99)
```


## Fractional Brownian Field (FBM) Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{2}\left(||\mathbf{x}||_{2}^{2H} + ||\mathbf{y}||_{2}^{2H} - ||\mathbf{x}-\mathbf{y}||_{2}^{2H}\right)
$$

```scala
val fbm = new FBMKernel(0.99)
```

The FBM kernel is the generalization of fractional Brownian motion to multi-variate index sets. Fractional Brownian motion is a stochastic process which is the generalization of Brownian motion, it was first studied by [Mandelbrot and Von Ness](https://www.jstor.org/stable/2027184). It is a _self similar_ stochastic process, with stationary increments. However the process itself is non-stationary (as can be seen from the expression for the kernel) and has long range non vanishing covariance.

## Maximum Likelihood Perceptron Kernel

The _maximum likelihood perceptron_ (MLP) kernel, was first arrived at in Radford Neal's [thesis](http://www.cs.toronto.edu/~radford/ftp/thesis.pdf), by considering the limiting case of a bayesian feed forward neural network with sigmoid activation.

$$
C(\mathbf{x},\mathbf{y}) = sin^{-1} \left (\frac{w \mathbf{x}^\intercal \mathbf{y} + b}{(w \mathbf{x}^\intercal \mathbf{x} + b) (w \mathbf{y}^\intercal \mathbf{y} + b)} \right )
$$

## Neural Network Kernel

Also a result of limiting case of bayesian neural networks, albeit with $erf(.)$ as the transfer function.

$$
C(\mathbf{x},\mathbf{y}) = \frac{2}{\pi} sin \left (\frac{2 \mathbf{x}^\intercal \Sigma \mathbf{y}}{(2 \mathbf{x}^\intercal \Sigma \mathbf{x} + 1) (2 \mathbf{y}^\intercal \Sigma \mathbf{y} + 1)} \right )
$$
