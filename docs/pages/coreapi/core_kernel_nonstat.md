---
title: Non-stationary Kernels
tags: [kernels,covariance_functions,non_stationary_kernels]
sidebar: coreapi_sidebar
permalink: core_kernel_nonstat.html
folder: coreapi
---

## Polynomial Kernel

A very popular non-stationary kernel used in machine learning, the polynomial represents the data features as polynomial expansions upto an index $$d$$.

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

{% include links.html %}
