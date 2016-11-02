---
title: Student T Processes
sidebar: coreapi_sidebar
permalink: core_stp.html
folder: coreapi
---

[Student T Processes](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf) (STP) can be viewed as a generalization of Gaussian Processes, in GP models we use the multivariate normal distribution to model noisy observations of an unknown function. Likewise for STP models, we employ the multivariate student t distribution. Formally a student t process is a stochastic process where the finite dimensional distribution is multivariate t.

$$
\begin{align}
\mathbf{y} & \in \mathbb{R}^n \\
\mathbf{y} & \sim MVT_{n}(\nu, \phi, K) \\
p(\mathbf{y}) & = \frac{\Gamma(\frac{\nu + n}{2})}{((\nu - 2)\pi)^{n/2} \Gamma(\nu/2)} |K|^{-1/2} \\
& \times (1 + (\mathbf{y} - \phi)^T K^{-1} (\mathbf{y} - \phi))^{-\frac{\nu +n}{2}}
\end{align}
$$

It is known that as $$\nu \rightarrow \infty$$, the $$MVT_{n}(\nu, \phi, K)$$ tends towards the multivariate normal distribution $$\mathcal{N}_{n}(\phi, K)$$.

## Regression with Student T Processes

The regression formulation for STP models is identical to the GP regression framework, to summarize the posterior predictive distribution takes the following form.

Suppose $$\mathbf{t} \sim MVT_{n_{tr} + n_t}(\nu, \mathbf{0}, K)$$ is the process producing the data.
Let $$[\mathbf{f_*}]_{n_{t} \times 1}$$ represent the values of the function on the test inputs and $$[\mathbf{y}]_{n_{tr} \times 1}$$ represent noisy observations made on the training data points.

$$
	\begin{align}
		& \mathbf{f_*}|X,\mathbf{y},X_* \sim MVT_{\nu + n_{tr}}(\mathbf{\bar{f_*}}, \frac{\nu + \beta - 2}{\nu + n_{tr} - 2} \times cov(\mathbf{f_*}))  \label{eq:posterior}\\
    & \beta = \mathbf{y}^T K^{-1} \mathbf{y} \\
		& \mathbf{\bar{f_*}} \overset{\triangle}{=} \mathbb{E}[\mathbf{f_*}|X,y,X_*] = K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1} \mathbf{y} \label{eq:posterior:mean} \\
		& cov(\mathbf{f_*}) = K(X_*,X_*) - K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1}K(X,X_*)

	\end{align}
$$

### STP models for a single output

For univariate GP models (single output), use the [```StudentTRegressionModel```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.stp.StudentTRegression) class (an extension of [```AbstractSTPRegressionModel```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.stp.AbstractSTPRegressionModel)). To construct a STP regression model you would need:


* The degrees of freedom $$\nu$$
* Kernel/covariance instance to model correlation between values of the latent function at each pair of input features.
* Kernel instance to model the correlation of the additive noise, generally the ```DiracKernel``` (white noise) is used.
* Training data

```scala
val trainingdata: Stream[(DenseVector[Double], Double)] = ...
val num_features = trainingdata.head._1.length

// Create an implicit vector field for the creation of the stationary
// radial basis function kernel
implicit val field = VectorField(num_features)

val kernel = new RBFKernel(2.5)
val noiseKernel = new DiracKernel(1.5)
val model = new StudentTRegression(1.5, kernel, noiseKernel, trainingData)
```

### STP models for Multiple Outputs

Working with multi-output Student T models is similar to [multi-output GP]({{site.baseurl}}/core_gp.html#gp-models-for-multiple-outputs) models. We need to create a kernel function over the combined index set ```(DenseVector[Double], Int)```. This can be done using the _sum of separable_ kernel idea.

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

You can use the [```MOStudentTRegression[I]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.models.gp.MOGPRegressionModel) class to create multi-output GP models.

```scala
val trainingdata: Stream[(DenseVector[Double], DenseVector[Double])] = ...

val model = new MOStudentTRegression[DenseVector[Double]](
          sos_kernel, sos_noise, trainingdata,
          trainingdata.length, trainingdata.head._2.length)

```
