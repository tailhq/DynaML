---
title: Stationary Kernels
tags: [kernels,covariance_functions,stationary_kernels]
sidebar: coreapi_sidebar
permalink: core_kernel_stat.html
folder: coreapi
---

Stationary kernels can be expressed as a function of the difference between their inputs.

$$
	C(\mathbf{x}, \mathbf{y}) = K(||\mathbf{x} - \mathbf{y}||_{p})
$$

Note that any norm may be used to quantify the distance between the two vectors $$\mathbf{x} \ \& \ \mathbf{y}$$. The values $$p = 1$$ and $$p = 2$$ represent the _Manhattan distance_ and _Euclidean distance_ respectively.

Stationary kernels are implemented as a subset of the [```StationaryKernel[T, V, M]```]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.StationaryKernel) class which requires a ```Field[T]``` implicit object (an algebraic field which has definitions for addition, subtraction, multiplication and division of its elements much like the number system). You may also import ```spire.implicits._``` in order to load the default field implementations for basic data types like ```Int```, ```Double``` and so on. Before instantiating any child class of ```StationaryKernel``` one needs to enter the following code.

```scala
import spire.algebra.Field
import io.github.mandar2812.dynaml.analysis.VectorField
//Calculate the number of input features
//and create a vector field of that dimension
val num_features: Int = ...
implicit val f = VectorField(num_features)
```

## Radial Basis Function Kernel / Squared Exponential Kernel

![kernel]({{site.baseurl}}/images/gaussiankernel.jpg)

$$
	C(\mathbf{x},\mathbf{y}) = e^{-\frac{1}{2}||\mathbf{x}-\mathbf{y}||_{2}^2/\sigma^2}
$$

The RBF kernel is the most popular kernel function applied in machine learning, it represents an inner product space which is spanned by the _Hermite_ polynomials and as such is suitable to model smooth functions. The RBF kernel is also called a _universal_ kernel for the reason that any smooth function can be represented with a high degree of accuracy assuming we can find a suitable value of the bandwidth.

```scala
val rbf = new RBFKernel(4.0)
```

A genralization of the RBF Kernel is the Squared Exponential Kernel

$$
	C(\mathbf{x},\mathbf{y}) = h e^{-\frac{||\mathbf{x}-\mathbf{y}||_{2}^2}{2l^2}}
$$

```scala
val rbf = new SEKernel(4.0, 2.0)
```

## Laplacian Kernel

$$
	C(\mathbf{x},\mathbf{y}) = e^{-\frac{1}{2}||\mathbf{x}-\mathbf{y}||_1/\beta}
$$

The Laplacian kernel is the covariance function of the well known [Ornstein Ulhenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), samples drawn from this process are continuous and only once differentiable.

```scala
val lap = new LaplacianKernel(4.0)
```

## T-Student Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{1 + ||\mathbf{x}-\mathbf{y}||^d}
$$

```scala
val tstud = new TStudentKernel(2.0)
```


## Rational Quadratic Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \left( 1 + \frac{||\mathbf{x}-\mathbf{y}||^2}{2 \mu \ell^2} \right)^{-\frac{1}{2}  (dim(\mathbf{x})+\mu)}
$$

```scala
val rat = new RationalQuadraticKernel(shape = 1.5, l = 1.5)
```

## Cauchy Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{1 + \frac{||\mathbf{x}-\mathbf{y}||^2}{\sigma^2}}
$$

```scala
val cau = new CauchyKernel(2.5)
```

## Wavelet Kernel

The Wavelet kernel ([Zhang et al, 2004](http://dx.doi.org/10.1109/TSMCB.2003.811113)) comes from Wavelet theory and is given as

$$
	C(\mathbf{x},\mathbf{y}) = \prod_{i = 1}^{d} h\left(\frac{x_i-y_i}{a}\right)
$$

Where the function `h` is known as the mother wavelet function, Zhang et. al suggest the following expression for the mother wavelet function.

$$
	h(x) = cos(1.75x)e^{-x^2/2}
$$

```scala
val wv = new WaveletKernel(x => math.cos(1.75*x)*math.exp(-1.0*x*x/2.0))(1.5)
```

{% include links.html %}
