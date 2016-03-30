---
layout: page
title: Kernels
---

-----

![kernel]({{site.baseurl}}/images/kernel.png)


_Positive definite_ functions or _positive type_ functions occupy an important place in various areas of mathematics, from the construction of covariances of random variables to quantifying distance measures in _Hilbert spaces_. Symmetric positive type functions defined on the cartesian product of a set with itself $$K: E \times E \rightarrow \mathbb{R}$$ are also known as _kernel_ functions in machine learning. They are applied extensively in problems such as.

1. Represent non-linear behavior in SVM models: [_SVM_](https://en.wikipedia.org/wiki/Support_vector_machine) and [_LSSVM_](http://www.worldscientific.com/worldscibooks/10.1142/5089)
2. Quantify covariance between input patterns: [_Gaussian Processes_](http://www.gaussianprocess.org/gpml/)
3. Represent degree of 'closeness' or affinity in unsupervised learning: [_Kernel Spectral Clustering_](http://arxiv.org/pdf/1505.00477.pdf)

For an in depth review of the various applications of kernels in the machine learning domain, refer to [Scholkopf et. al](http://www.kernel-machines.org/publications/pdfs/0701907.pdf)


## Kernels available in DynaML


### Radial Basis Function Kernel / Squared Exponential Kernel

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

### Laplacian Kernel

$$
	C(\mathbf{x},\mathbf{y}) = e^{-\frac{1}{2}||\mathbf{x}-\mathbf{y}||_1/\beta}
$$

The Laplacian kernel is the covariance function of the well known [Ornstein Ulhenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), samples drawn from this process are continuous and only once differentiable.

```scala
val lap = new LaplacianKernel(4.0)
```

### T-Student Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{1 + ||\mathbf{x}-\mathbf{y}||_{2}^d}
$$

```scala
val tstud = new TStudentKernel(2.0)
```


### Rational Quadratic Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{||\mathbf{x}-\mathbf{y}||_{2}^2}{c + ||\mathbf{x}-\mathbf{y}||_{2}^2}
$$

```scala
val rat = new RationalQuadraticKernel(3.5)
```

### Cauchy Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{1 + \frac{||\mathbf{x}-\mathbf{y}||_{2}^2}{\sigma^2}}
$$

```scala
val cau = new CauchyKernel(2.5)
```

### Wavelet Kernel

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


### Fractional Brownian Field (FBM) Kernel

$$
	C(\mathbf{x},\mathbf{y}) = \frac{1}{2}\left(||\mathbf{x}||_{2}^{2H} + ||\mathbf{y}||_{2}^{2H} - ||\mathbf{x}-\mathbf{y}||_{2}^{2H}\right)
$$

```scala
val fbm = new FBMKernel(0.99)
```

The FBM kernel is the generalization of fractional Brownian motion to multi-variate index sets. Fractional Brownian motion is a stochastic process which is the generalization of Brownian motion, it was first studied by [Mandelbrot and Von Ness](https://www.jstor.org/stable/2027184). It is a _self similar_ stochastic process, with stationary increments. However the process itself is non-stationary (as can be seen from the expression for the kernel) and has long range non vanishing covariance.

-----

## Creating Composite Kernels

In machine learning it is well known that kernels can be combined to give other valid kernels. The symmetric positive semi-definite property of a kernel is preserved as long as it is added or multiplied to another valid kernel. In DynaML adding and multiplying kernels is elementary.

```scala

val k1 = new RBFKernel(2.5)
val k2 = new RationalQuadraticKernel(2.0)

val k = k1 + k2
```

-----

## Implementing Custom Kernels

For more details on implementing user defined kernels, refer to the [wiki](https://github.com/mandar2812/DynaML/wiki/Kernels).
