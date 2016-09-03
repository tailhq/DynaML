---
title: Gaussian Processes
sidebar: coreapi_sidebar
permalink: core_gp.html
folder: coreapi
---

## Gaussian Process Regression

$$
	\begin{align}
		& y = f(x) + \epsilon \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
		& \left(\mathbf{y} \ \ \mathbf{f_*} \right)^T \sim \mathcal{N}\left(\mathbf{0}, \left[ \begin{matrix} K(X, X) + \sigma^{2} \it{I} & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{matrix} \right ] \right)

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
