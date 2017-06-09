!!! summary
    The _Generalized Least Squares_ model is a regression formulation which does not assume that the model errors/residuals are independent of each other. Rather it borrows from the _Gaussian Process_ paradigm and assigns a covariance structure to the model residuals.


!!! warning
    The nomenclature _Generalized Least Squares_ (GLS) and _Generalized Linear Models_ (GLM) can cause much confusion. It is important to remember the context of both. GLS refers to relaxing of the independence of residuals assumption while GLM refers to _Ordinary Least Squares_ OLS based models which are extended to model regression, counts, or classification tasks.


## Formulation.

Let $\mathbf{X} \in \mathbb{R}^{n\times m}$ be a matrix containing data attributes. The GLS model builds a linear predictor of the target quantity of the following form.

$$
\begin{equation}
\mathbf {y} = \varphi(\mathbf {X}) \mathbf {\beta } +\mathbf {\varepsilon }
\end{equation}
$$

Where $\varphi(.): \mathbb{R}^m \rightarrow \mathbb{R}^d$ is a feature mapping, $\mathbf{y} \in \mathbb{R}^n$ is the vector of output values found in the training data set and $\mathbf{\beta} \in \mathbb{R}^d$ is a set of regression parameters.

In the GLS framework, it is assumed that the model errors $\varepsilon \in \mathbb{R}^n$ follow a multivariate gaussian distribution given by $\mathbb {E} [\varepsilon |\mathbf {X} ] = 0$ and $\operatorname{Var} [\varepsilon |\mathbf {X} ] = \mathbf {\Omega }$, where $\mathbf{\Omega}$ is a symmetric positive semi-definite covariance matrix.

In order to calculate the model parameters $\mathbf{\beta}$, the log-likelihood of the training data outputs must be maximized with respect to the parameters $\mathbf{\beta}$, which leads to.

$$
\begin{equation}
\min_{\mathbf{\beta}} \ \mathcal{J}_P(\mathbf{\beta}) = (\mathbf {y} - \varphi(\mathbf {X}) \mathbf {\beta} )^{\mathtt {T}}\,\mathbf {\Omega } ^{-1}(\mathbf {y} - \varphi(\mathbf {X}) \mathbf {\beta} )
\end{equation}
$$

For the GLS problem the analytical solution of the above optimization problem can be calculated.

$$
{\displaystyle \mathbf {\hat {\beta }} =\left(\varphi(\mathbf {X}) ^{\mathtt {T}}\mathbf {\Omega } ^{-1}\varphi(\mathbf {X})\right)^{-1}\varphi(\mathbf {X}) ^{\mathtt {T}}\mathbf {\Omega } ^{-1}\mathbf {y} .}
$$

## GLS Models

You can create a GLS model using the [`#!scala GeneralizedLeastSquaresModel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.models.lm.GeneralizedLeastSquaresModel) class.

```scala
//Get the training data
val data: Stream[(DenseVector[Double], Double)] = _
//Define a feature mapping
//If it is not defined the GLS model
//will assume a identity feature map.
val feature_map: DenseVector[Double] => DenseVector[Double] = _

//Initialize a kernel function.
val kernel: LocalScalarKernel[DenseVector[Double]] = _
//Construct the covariance matrix for model errors.
val covmat = kernel.buildBlockedKernelMatrix(data, data.length)

val gls_model = new GeneralizedLeastSquaresModel(
  data, covmat,
  feature_map)

//Train the model
gls_model.learn()
```
