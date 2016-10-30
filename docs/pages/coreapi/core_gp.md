---
title: Gaussian Processes
sidebar: coreapi_sidebar
permalink: core_gp.html
folder: coreapi
---

_Gaussian Process_ models are well supported in DynaML, the [```AbstractGPRegressionModel[T, I]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel) and [```AbstractGPClassification[T, I]```](/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.models.gp.AbstractGPClassification) classes which extend the [```StochasticProcess[T, I, Y, W]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.StochasticProcess) base trait are the starting point for all GP implementations.

The ```StochasticProcess[T, I, Y, W]``` trait contains the ```predictiveDistribution[U <: Seq[I]](test: U): W``` method which returns the posterior predictive distribution (represented by the generic type ```W```).

The base trait is extended by [```SecondOrderProcess[T, I, Y, K, M, W]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.SecondOrderProcess) which defines a skeleton for processes which are defined by their first and second order statistics (_mean functions_ and _covariance functions_).

Since for most applications it is assumed that the training data is standardized, the mean function is often chosen to be zero $$\mu(\mathbf{x}) = 0$$, thus the covariance function or kernel defines all the interesting behavior of _second order processes_. For a more in depth information on the types of covariance functions available visit the [kernels]({{site.baseurl}}/core_kernels.html) page.


## Gaussian Process Regression

The GP regression framework aims to infer an unknown function $$f(x)$$ given $$y_i$$ which are noise corrupted observations of this unknown function. This is done by adopting an explicit probabilistic formulation to the multi-variate distribution of the noise corrupted observations $$y_i$$ conditioned on the input features (or design matrix) $$X$$

$$
	\begin{align}
		& y = f(x) + \epsilon \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
		& \left(\mathbf{y} \ \ \mathbf{f_*} \right)^T | X \sim \mathcal{N}\left(\mathbf{0}, \left[ \begin{matrix} K(X, X) + \sigma^{2} \it{I} & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{matrix} \right ] \right)

	\end{align}
$$


In the presence of training data

$$
	X = (x_1, x_2, \cdot , x_n) \ \mathbf{y} = (y_1, y_2, \cdot , y_n)
$$

Inference is carried out by calculating the posterior predictive distribution over the unknown targets

$$
	\mathbf{f_*}|X,\mathbf{y},X_*
$$

assuming $$ X_* $$, the test inputs are known.

$$
	\begin{align}
		& \mathbf{f_*}|X,\mathbf{y},X_* \sim \mathcal{N}(\mathbf{\bar{f_*}}, cov(\mathbf{f_*}))  \label{eq:posterior}\\
		& \mathbf{\bar{f_*}} \overset{\triangle}{=} \mathbb{E}[\mathbf{f_*}|X,y,X_*] = K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1} \mathbf{y} \label{eq:posterior:mean} \\
		& cov(\mathbf{f_*}) = K(X_*,X_*) - K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1}K(X,X_*)

	\end{align}
$$

### GP models for a single output

For univariate GP models (single output), use the [```GPRegression```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.gp.GPRegressionModel) class (an extension of ```AbstractGPRegressionModel```). To construct a GP regression model you would need:

* Training data
* Kernel/covariance instance to model correlation between values of the latent function at each pair of input features.
* Kernel instance to model the correlation of the additive noise, generally the ```DiracKernel``` (white noise) is used.

```scala
val trainingdata: Stream[(DenseVector[Double], Double)] = ...
val num_features = trainingdata.head._1.length

// Create an implicit vector field for the creation of the stationary
// radial basis function kernel
implicit val field = VectorField(num_features)

val kernel = new RBFKernel(2.5)
val noiseKernel = new DiracKernel(1.5)
val model = new GPRegression(kernel, noiseKernel, trainingData)
```

### GP models for multiple outputs

As reviewed in [Lawrence et.al](https://arxiv.org/abs/1106.6251), Gaussian Processes for multiple outputs can be interpreted as single output GP models with an expanded index set. Recall that GPs are stochastic processes and thus are defined on some _index set_, for example in the equations above it is noted that $$x \in \mathbb{R}^p$$ making $$\mathbb{R}^p$$ the _index set_ of the process.

In case of multiple outputs the index set is expressed as a cartesian product $$x \in \mathbb{R}^{p} \times \{1,2, \cdots, d \}$$, where $$d$$ is the number of outputs to be modeled.

It needs to be noted that now we will also have to define the kernel function on the same index set i.e. $$\mathbb{R}^{p} \times \{1,2, \cdots, d \}$$.

In multi-output GP literature a common way to construct kernels on such index sets is to multiply base kernels on each of the parts $$\mathbb{R}^p$$ and $$\{1,2,\cdots,d\}$$, such kernels are known as _separable kernels_.

$$
\begin{equation}
K((\mathbf{x}, d), (\mathbf{x}', d')) = K_{x}(\mathbf{x}, \mathbf{x}') . K_{d}(d, d')
\end{equation}
$$

Taking this idea further _sum of separable kernels_ (SoS) are often employed in multi-output GP models. These models are also known as _Linear Models of Co-Regionalization_ (LMC) and the kernels which encode correlation between the outputs $$K_d(.,.)$$ are known as _co-regionalization kernels_.

$$
\begin{equation}
K((\mathbf{x}, d), (\mathbf{x}', d')) = \sum_{i = 1}^{D} K^{i}_{x}(\mathbf{x}, \mathbf{x}') . K^{i}_{d}(d, d')
\end{equation}
$$

Creating such SoS kernels in DynaML is quite straightforward, use the ```:*``` operator to multiply a kernel defined on ```DenseVector[Double]``` with a kernel defined on ```Int```.

```scala

val linearK = new PolynomialKernel(2, 1.0)
val tKernel = new TStudentKernel(0.2)
val d = new DiracKernel(0.037)

val mixedEffects = new MixedEffectRegularizer(0.5)
val coRegCauchyMatrix = new CoRegCauchyKernel(10.0)
val coRegDiracMatrix = new CoRegDiracKernel

val sos_kernel: CompositeCovariance[(DenseVector[Double], Int)] =
  (linearK :* mixedEffects)  + (tKernel :* coRegCauchyMatrix)

val sos_noise: CompositeCovariance[(DenseVector[Double], Int)] = d :* coRegDiracMatrix

```

You can use the [```MOGPRegressionModel[I]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.gp.MOGPRegressionModel) class to create multi-output GP models.

```scala
val trainingdata: Stream[(DenseVector[Double], DenseVector[Double])] = ...

val model = new MOGPRegressionModel[DenseVector[Double]](
          sos_kernel, sos_noise, trainingdata,
          trainingdata.length, trainingdata.head._2.length)

```

<br/>

## Gaussian Process Binary Classification

Gaussian process models for classification are formulated using two components.

- A latent (nuisance) function $$f(x)$$
- A transfer function $$\sigma(.)$$ which transforms the value $$f(x)$$ to a class probability

$$
	\begin{align}
		& \pi(x) \overset{\triangle}{=} p(y = +1| x) = \sigma(f(x)) \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
	\end{align}
$$

Inference is divided into two steps.

- Computing the distribution of the latent function corresponding to a test case

$$
\begin{align}
	& p(f_*|X, \mathbf{y}, x_*) = \int p(f_*|X, \mathbf{y}, x_*, \mathbf{f}) p(\mathbf{f}|X, \mathbf{y}) d\mathbf{f} \\
	& p(\mathbf{f}|X, \mathbf{y}) = p(\mathbf{y}| \mathbf{f}) p(\mathbf{f}|X)/ p(\mathbf{y}|X)
\end{align}

$$

- Generating probabilistic prediction for a test case.

$$
\bar{\pi_*} \overset{\triangle}{=} p(y_* = +1| X, \mathbf{y}, x_*) = \int \sigma(f_*) p(f_*|X, \mathbf{y}, x_*) df_*
$$

```scala
val trainingdata: Stream[(DenseVector[Double], Double)] = ...
val num_features = trainingdata.head._1.length

// Create an implicit vector field for the creation of the stationary
// radial basis function kernel
implicit val field = VectorField(num_features)

val kernel = new RBFKernel(2.5)
val likelihood = new VectorIIDSigmoid()
val model = new LaplaceBinaryGPC(trainingData, kernel, likelihood)
```


To learn more about extending the Gaussian Process base classes/traits refer to the [wiki](https://github.com/mandar2812/DynaML/wiki/Gaussian-Processes).
