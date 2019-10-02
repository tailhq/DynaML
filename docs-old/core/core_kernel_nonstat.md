Non-stationary covariance functions cannot be expressed as simply a function of the distance between their inputs $\mathbf{x} - \mathbf{y}$.

## Locally Stationary Kernels

A simple way to construct non-stationary covariances from stationary ones is by scaling the original stationary covariance; $K(\mathbf{x} - \mathbf{y})$, by a function of $\mathbf{x} + \mathbf{y}$.

$$
C(\mathbf{x}, \mathbf{y}) = G(\mathbf{x} + \mathbf{y}) K(\mathbf{x} - \mathbf{y})
$$

Here $G(.): \mathcal{X} \rightarrow \mathbb{R}$ is a non-negative function of its inputs. These kernels are called _locally stationary kernels_. For an in-depth review of locally stationary kernels refer to [Genton et. al](http://jmlr.csail.mit.edu/papers/volume2/genton01a/genton01a.pdf).

```scala
//Instantiate the base kernel
val kernel: LocalScalarKernel[I] = _

val scalingFunction: (I) => Double = _

val scKernel = new LocallyStationaryKernel(
	kernel, DataPipe(scalingFunction))
```

## Polynomial Kernel

A very popular non-stationary kernel used in machine learning, the polynomial represents the data features as polynomial expansions up to an index $d$.

$$
C(\mathbf{x},\mathbf{y}) = (\mathbf{x}^\intercal \mathbf{y} + a)^{d}
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
