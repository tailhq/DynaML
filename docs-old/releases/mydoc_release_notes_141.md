---
title: Release notes 1.4.1
tags: [getting_started,release_notes]
keywords: release notes, announcements, what's new, new features
last_updated: March 26, 2017
summary: ""
sidebar: mydoc_sidebar
permalink: mydoc_release_notes_141.html
folder: mydoc
---

!!! summary ""
    Version 1.4.1 of DynaML, released March 26, 2017, implements a number of new models (Extended Skew GP, student T process, generalized least squares, etc) and features.


## Pipes API

### Additions

The pipes API has been vastly extended by creating pipes which encapsulate functions of multiple arguments leading to the following end points.

 - `#!scala DataPipe2[A, B, C]`: Pipe which takes 2 arguments
 - `#!scala DataPipe3[A, B, C, D]` : Pipe which takes 3 arguments
 - `#!scala DataPipe4[A, B, C, D, E]`: Pipe which takes 4 arguments

Furthermore there is now the ability to create pipes which return pipes, something akin to curried functions in functional programming.

 - `#!scala MetaPipe`: Takes an argument returns a `DataPipe`
 - `#!scala MetaPipe21`: Takes 2 arguments returns a `DataPipe`
 - `#!scala MetaPipe12`: Takes an argument returns a `DataPipe2`


 A new kind of Stream data pipe, `#!scala StreamFlatMapPipe` is added to represent data pipelines which can perform flat map like operations on streams.


```scala
val mapFunc: (I) => Stream[J] = ...
val streamFMPipe = StreamFlatMapPipe(mapFunc)
```

 - Added Data Pipes API for Apache Spark RDDs.

```scala
val num = 20
val numbers = sc.parallelize(1 to num)
val convPipe = RDDPipe((n: Int) => n.toDouble)

val sqPipe = RDDPipe((x: Double) => x*x)

val sqrtPipe = RDDPipe((x: Double) => math.sqrt(x))

val resultPipe = RDDPipe((r: RDD[Double]) => r.reduce(_+_).toInt)

val netPipeline = convPipe > sqPipe > sqrtPipe > resultPipe
netPipeline(numbers)
```

 - Added `#!scala UnivariateGaussianScaler` class for gaussian scaling of univariate data.




## Core API

### Additions


**Package** `#!scala dynaml.models.bayes`

This new package will house stochastic prior models, currently there is support for GP and Skew GP priors, to see a starting example see `stochasticPriors.sc` in the `scripts` directory of the DynaML source.

----

**Package** `#!scala dynaml.kernels`

 - Added `#!scala evaluateAt(h)(x,y)` and `#!scala gradientAt(h)(x,y)`; expressing `#!scala evaluate(x,y)` and `#!scala gradient(x,y)` in terms of them
 - Added `#!scala asPipe` method for Covariance Functions
 - For backwards compatibility users are advised to extend
   `#!scala LocalSVMKernel` in their custom Kernel implementations incase they do
   not want to implement the `#!scala evaluateAt` API endpoints.
 - Added [`FeatureMapKernel`](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/models/lm/SparkLogisticGLM.scala), representing kernels which can be explicitly decomposed into feature mappings.
 - Added Matern half integer kernel [`#!scala GenericMaternKernel[I]`](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/kernels/GenericMaternKernel.scala)
 - Added `#!scala block(S: String*)` method to block any hyper-parameters of kernels.
 - Added `#!scala NeuralNetworkKernel` and `#!scala GaussianSpectralKernel`.
 - Added `#!scala DecomposableCovariance`

```scala
import io.github.tailhq.dynaml.DynaMLPipe._
import io.github.tailhq.dynaml.kernels._


implicit val ev = VectorField(6)
implicit val sp = breezeDVSplitEncoder(2)
implicit val sumR = sumReducer

val kernel = new LaplacianKernel(1.5)
val other_kernel = new PolynomialKernel(1, 0.05)

val decompKernel = new DecomposableCovariance(kernel, other_kernel)(sp, sumReducer)

val other_kernel1 = new FBMKernel(1.0)

val decompKernel1 = new DecomposableCovariance(decompKernel, other_kernel1)(sp, sumReducer)

val veca = DenseVector.tabulate[Double](8)(math.sin(_))
val vecb = DenseVector.tabulate[Double](8)(math.cos(_))

decompKernel1.evaluate(veca, vecb)

```

----

**Package** `#!scala dynaml.algebra`

Partitioned Matrices/Vectors and the following operations

  - Addition, Subtraction
  - Matrix, vector multiplication
  - LU, Cholesky
  - `#!scala A\y`, `#!scala A\Y`

Added calculation of quadratic forms, namely:

  - `#!scala quadraticForm` which calculates $\mathbf{x}^\intercal A^{-1} \mathbf{x}$
  - `#!scala crossQuadraticForm` which calculates $\mathbf{y}^\intercal A^{-1} \mathbf{x}$

Where A is assumed to be a symmetric positive semi-definite matrix

Usage:

```scala
import io.github.tailhq.dynaml.algebra._

val x: DenseVector[Double] = ...
val y: DenseVector[Double] = ...
val a: DenseMatrix[Double] = ...

quadraticForm(a,x)
crossQuadraticForm(y, a, x)

```

----

**Package** `#!scala dynaml.modelpipe`

New package created, moved all inheriting classes of `#!scala ModelPipe` to this package.

Added the following:

 - `#!scala GLMPipe2` A pipe taking two arguments and returning a `#!scala GeneralizedLinearModel` instance
 - `#!scala GeneralizedLeastSquaresPipe2`:
 - `#!scala GeneralizedLeastSquaresPipe3`:

----

**Package** `#!scala dynaml.models`

 - Added a new Neural Networks API: `#!scala NeuralNet` and `#!scala GenericFFNeuralNet`, for an example refer to `#!scala TestNNDelve` in `#!scala dynaml-examples`.
 - `#!scala GeneralizedLeastSquaresModel`: The  [GLS](https://en.wikipedia.org/wiki/Generalized_least_squares) model.
 - `#!scala ESGPModel`: The implementation of a skew gaussian process regression model
 - [Warped Gaussian Process](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/models/gp/WarpedGPModel.scala) models **WIP**
 - Added mean function capability to Gaussian Process and Student T process models.
 - Added Apache Spark implementation of Generalized Linear Models; see [SparkGLM](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/models/lm/SparkGLM.scala), [SparkLogisticModel](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/models/lm/SparkLogisticGLM.scala), [SparkProbitGLM](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/models/lm/SparkLogisticGLM.scala)

 ----

**Package** `dynaml.probability`

 - `#!scala MultivariateSkewNormal` as specified in [Azzalani et. al](https://arxiv.org/pdf/0911.2093.pdf)
 - `#!scala ExtendedMultivariateSkewNormal`
 - `#!scala UESN` and `#!scala MESN` representing an alternative formulation of the skew gaussian family from Adcock and Shutes.
 - `#!scala TruncatedGaussian`: Truncated version of the Gaussian distribution.
 - [Matrix Normal Distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
 - Added _Expectation_ operator for `#!scala RandomVariable` implementations in the `#!scala io.github.tailhq.dynaml.probability` package object. Usage example given below.
 - `#!scala SkewGaussian`, `#!scala ExtendedSkewGaussian`: An breeze implementation of the SkewGaussian and extended Skew-Gaussian distributions respectively
 - `#!scala PushforwardMap`, `#!scala DifferentiableMap` added: `#!scala PushforwardMap` enables creating new random variables with defined density from base random variables.

``` scala
import io.github.tailhq.dynaml.analysis._
import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.probability.distributions._

val g = GaussianRV(0.0, 0.25)
val sg = RandomVariable(SkewGaussian(1.0, 0.0, 0.25))

//Define a determinant implementation for the Jacobian type (Double in this case)
implicit val detImpl = identityPipe[Double]

//Defines a homeomorphism y = exp(x) x = log(y)
val h: PushforwardMap[Double, Double, Double] = PushforwardMap(
  DataPipe((x: Double) => math.exp(x)),
  DifferentiableMap(
    (x: Double) => math.log(x),
    (x: Double) => 1.0/x)
)

//Creates a log-normal random variable
val p = h->g

//Creates a log-skew-gaussian random variable
val q = h->sg

//Calculate expectation of q
println("E[Q] = "+E(q))
```
 - Added _Markov Chain Monte Carlo_ (MCMC) based inference schemes `#!scala ContinuousMCMC` and the underlying sampling implementation in `#!scala GeneralMetropolisHastings`.
 - Added implementation of _Approximate Bayesian Computation_ (ABC) in the `#!scala ApproxBayesComputation` class.


``` scala
//The mean
val center: DenseMatrix[Double] = ...
//Covariance (positive semi-def) matrix among rows
val sigmaRows: DenseMatrix[Double] = ...
//Covariance (positive semi-def) matrix among columns
val sigmaCols: DenseMatrix[Double] = ...
val matD = MatrixNormal(center, sigmaRows, sigmaCols)
```
 - [Matrix T Distribution](https://en.wikipedia.org/wiki/Matrix_t-distribution) (_Experimental_)

``` scala
//The degrees of freedom (must be > 2.0 for existence of finite moments)
val mu: Double = ...
//The mean
val center: DenseMatrix[Double] = ...
//Covariance (positive semi-def) matrix among rows
val sigmaRows: DenseMatrix[Double] = ...
//Covariance (positive semi-def) matrix among columns
val sigmaCols: DenseMatrix[Double] = ...
val matD = MatrixT(mu, center, sigmaCols, sigmaRows)
```

----

**Package** `#!scala dynaml.optimization`

 - Added `#!scala ProbGPCommMachine` which performs grid search or CSA and then instead of selecting a single hyper-parameter configuration calculates a weighted Gaussian Process committee where the weights correspond to probabilities or confidence on each model instance (hyper-parameter configuration).

**Package** `#!scala dynaml.utils`

 - Added [multivariate gamma function](https://en.wikipedia.org/wiki/Multivariate_gamma_function)

``` scala
//Returns logarithm of multivariate gamma function
val g = mvlgamma(5, 1.5)
```

----

**Package** `#!scala dynaml.dataformat`

 - Added support for reading _MATLAB_ `.mat` files in the [`MAT`](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/dataformat/MAT.scala) object.


### Improvements/Bug Fixes

**Package** `#!scala dynaml.probability`

 - Removed `#!scala ProbabilityModel` and replaced with `#!scala JointProbabilityScheme` and `#!scala BayesJointProbabilityScheme`, major refactoring to `#!scala RandomVariable` API.

----

**Package** `#!scala dynaml.optimization`

 - Improved logging of `#!scala CoupledSimulatedAnnealing`
 - Refactored `#!scala GPMLOptimizer` to `#!scala GradBasedGlobalOptimizer`

----

**Package** `#!scala dynaml.utils`
 - Correction to `#!scala utils.getStats` method used for calculating mean and variance of data sets consisting of `#!scala DenseVector[Double]`.
 - `#!scala minMaxScalingTrainTest` `#!scala minMaxScaling` of `#!scala DynaMLPipe` using `#!scala GaussianScaler` instead of `#!scala MinMaxScaler` for processing of features.

----

**Package** `#!scala dynaml.kernels`

 - Fix to `#!scala CoRegCauchyKernel`: corrected mismatch of hyper-parameter string
 - Fix to `#!scala SVMKernel` objects matrix gradient computation in the case when kernel dimensions are not multiples of block size.
 - Correction to gradient calculation in RBF kernel family.
 - Speed up of kernel gradient computation, kernel and kernel gradient matrices with respect to the model hyper-parameters now calculated in a single pass through the data.
