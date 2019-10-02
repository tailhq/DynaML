!!! summary
    The multi-output matrix T regression model was first described by [Conti and O' Hagan](http://www.sciencedirect.com/science/article/pii/S0378375809002559) in their paper on Bayesian emulation of multi-output computer codes. It has been available in the [`dynaml.models.stp`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.models.stp.package) package of the `dynaml-core` module since [v1.4.2](/releases/mydoc_release_notes_142.md#additions).


## Formulation

The model starts from the multi-output gaussian process framework. The quantity of interest is some unknown function $\mathbf{f}: \mathcal{X} \rightarrow \mathbb{R}^q$, which maps inputs in $\mathcal{X}$ (an arbitrary input space) to a $q$ dimensional vector outputs.

$$
\begin{align}
\mathbf{f}(.)|B,\Sigma, \theta &\sim \mathcal{GP}(\mathbf{m}(.), c(.,.)\Sigma) \\
\mathbf{m}(x) &= B^\intercal \varphi(x)
\end{align}
$$

The input $x$ is transformed through $\varphi(.): \mathcal{X} \rightarrow \mathbb{R}^m$ which is a deterministic feature mapping which then calculates the inputs for a linear _mean function_ $\mathbf{m}(.)$. The parameters of this linear trend are contained in the matrix $B \in \mathbb{R}^{m \times q}$ and $\theta$ contains all the covariance function hyper-parameters.

The prior distribution of the multi-output function is represented as a matrix normal distribution, with $c(.,.)$ representing the covariance between two input points, and the entries of $\Sigma$ being the covariance between the output dimensions.

The predictive distribution when the output data $D \in \mathbb{R}^{n\times q}$ is observed is calculated by first computing the conditional predictive distribution of $\mathbf{f}(.) | D, \Sigma, B, \theta$ and then integrating this distribution with respect to the posterior distributions $\Sigma|D$ and $B|D$.

The resulting predictive distribution $\mathbf{f}(.)| \theta, D$ has the following structure.

$$
\begin{align}
\mathbf{f}(.)|\theta,D &\sim \mathcal{T}(\mathbf{m}^{**}(.), c^{**}\Sigma_{GLS};n-m) \\
\end{align}
$$

The distribution is a [matrix variate T distribution](https://en.wikipedia.org/wiki/Matrix_t-distribution). It is described by

  * Mean $\mathbf{m}^{**}(x)$.
  * Covariance between rows $c^{**}(x_{1}, x_{2})$
  * Covariance function between output columns $\Sigma_{GLS}$
  * Degrees of freedom $n-m$.

\begin{align}
\mathbf{m}^{**}(x_{1}) &= B_{GLS}^{\intercal}\varphi(x_{1}) + (D-\varphi(X)B_{GLS})^{\intercal} C^{-1}c(x_{1},.)\\
c^{**}(x_{1}, x_{2}) &= \bar{c}(x_{1}, x_{2}) + \hat{c}(x_{1}, x_{2})\\
\bar{c}(x_{1}, x_{2}) &= c(x_{1}, x_{2}) - C(x_{1},.)^{\intercal}C^{-1}C(x_{2},.) \\
\hat{c}(x_{1}, x_{2}) &= H(x_{1})^{\intercal}.A^{-1}.H(x_{2})\\
H(x) &= (\varphi(x) - \varphi(X)C^{-1}c(x,.)) \\
A &= \varphi(X)^{\intercal}C^{-1}\varphi(X)\\
\end{align}

The matrices $B_{GLS} = (\varphi(X)^{\intercal}C^{-1}\varphi(X))^{-1}\varphi(X)^{\intercal}C^{-1}D$ and $\Sigma_{GLS} = (n-m)^{-1}(D - \varphi(X)B_{GLS})^{\intercal}C^{-1}(D - \varphi(X)B_{GLS})$ are the [_generalized least squares_](/core/core_gls.md) estimators for the matrices $B$ and $\Sigma$ which we saw in the formulation above.

## Multi-output Regression

An implementation of the multi-output matrix T model is available via the class [`#!scala MVStudentsTModel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.models.stp.MVStudentsTModel). Instantiating the model is very similar to other stochastic process models in DynaML i.e. by specifying the covariance structures on signal and noise, training data, etc.

```scala
//Obtain the data, some generic type
val trainingdata: DataType = _
val num_data_points: Int = _
val num_outputs:Int = _

val kernel: LocalScalarKernel[I] = _
val noiseKernel: LocalScalarKernel[I] = _
val feature_map: DataPipe[I, Double] = _

//Define how the data is converted to a compatible type
implicit val transform: DataPipe[DataType, Seq[(I, Double)]] = _

val model = MVStudentsTModel(
  kernel, noiseKernel, feature_map)(
  trainingData, num_data_points, num_outputs)
```
