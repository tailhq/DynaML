!!! summary ""
    Summarizes some of the pet projects being tackled in DynaML

The past year has seen DynaML grow by leaps and bounds, this post hopes to give you an update about what has been achieved
and a taste for what is to come.

## Completed Features

A short tour of the enhancements which were completed.

### January to June

- Released `v1.3.x` series with the following new additions

  _Models_

  * Regularized Least Squares
  * Logistic and Probit Regression
  * Feed Forward Neural Nets
  * Gaussian Process (GP) classification and NARX based models
  * Least Squares Support Vector Machines (LSSVM) for classification and regression
  * Meta model API, committee models

  _Optimization Primitives_

  * Regularized Least Squares Solvers
  * Gradient Descent
  * Committee model solvers
  * Linear Solvers for LSSVM
  * Laplace approximation for GPs

  _Miscellaneous_

  * Data Pipes API
  * Migration to scala version 2.11.8

- Started work on release `1.4.x` series with initial progress

  _Improvements_

  * Migrated from Maven to Sbt.
  * Set [Ammonite](http://www.lihaoyi.com/Ammonite/) as default REPL.

### June to December

- Released `v1.4` with the following features.

  _Models_

  The following inference models have been added.

  * LSSVM committees.
  * Multi-output, multi-task _Gaussian Process_ models as reviewed in [Lawrence et. al](https://arxiv.org/abs/1106.6251).
  * _Student T Processes_: single and multi output inspired from [Shah, Ghahramani et. al](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
  * Performance improvement to computation of _marginal likelihood_ and _posterior predictive distribution_ in Gaussian Process models.
  * Posterior predictive distribution outputted by the ```AbstractGPRegression``` base class is now changed to ```MultGaussianRV``` which is added to the ```dynaml.probability``` package.

  _Kernels_

  * Added ```StationaryKernel``` and ```LocallyStationaryKernel``` classes in the kernel APIs, converted ```RBFKernel```, ```CauchyKernel```, ```RationalQuadraticKernel``` & ```LaplacianKernel``` to subclasses of ```StationaryKernel```

  * Added ```MLPKernel``` which implements the _maximum likelihood perceptron_ kernel as shown [here](http://gpss.cc/gpuqss16/slides/gp_gpss16_session2.pdf).

  * Added _co-regionalization kernels_ which are used in [Lawrence et. al](https://arxiv.org/abs/1106.6251) to formulate kernels for vector valued functions. In this category the following co-regionalization kernels were implemented.
    - ```CoRegRBFKernel```
    - ```CoRegCauchyKernel```
    - ```CoRegLaplaceKernel```
    - ```CoRegDiracKernel```

    * Improved performance when calculating kernel matrices for composite kernels.

    * Added ```:*``` operator to kernels so that one can create separable kernels used in _co-regionalization models_.

    _Optimization_

    * Improved performance of ```CoupledSimulatedAnnealing```, enabled use of 4 variants of _Coupled Simulated Annealing_, adding the ability to set annealing schedule using so called _variance control_ scheme as outlined in [de-Souza, Suykens et. al](ftp://ftp.esat.kuleuven.be/sista/sdesouza/papers/CSA2009accepted.pdf).

    _Pipes_

    * Added ```Scaler``` and ```ReversibleScaler``` traits to represent transformations which input and output into the same domain set, these traits are extensions of ```DataPipe```.

    * Added _Discrete Wavelet Transform_ based on the _Haar_ wavelet.

- Started work on `v1.4.1` with the following progress

  _Linear Algebra API_

  * Partitioned Matrices/Vectors and the following operations

    1. Addition, Subtraction
    2. Matrix, vector multiplication
    3. LU, Cholesky
    4. A\y, A\Y

  _Probability API_

  * Added API end points for representing Measurable Functions of random variables.

  _Model Evaluation_

  * Added Matthews Correlation Coefficient calculation to `BinaryClassificationMetrics` via the `matthewsCCByThreshold` method  

  _Data Pipes API_

  * Added `Encoder[S,D]` traits which are reversible data pipes representing an encoding between types `S` and `D`.

  _Miscellaneous_

  1. Updated ```ammonite``` version to `0.8.1`
  2. Added support for compiling basic R code with [renjin](http://www.renjin.org). Run R code in the following manner:

```scala
val toRDF = csvToRDF("dfWine", ';')
val wine_quality_red = toRDF("data/winequality-red.csv")
//Descriptive statistics
val commands: String = """
print(summary(dfWine))
print("\n")
print(str(dfWine))
"""
r(commands)
//Build Linear Model
val modelGLM = rdfToGLM("model", "quality", Array("fixed.acidity", "citric.acid", "chlorides"))
modelGLM("dfWine")
//Print goodness of fit
r("print(summary(model))")
```

## Ongoing Work

Some projects being worked on right now are.

* Bayesian optimization using Gaussian Process models.
* Implementation of Neural Networks using the [akka](http://akka.io) actor API.
* Implementation of kernels which can be decomposed on data dimensions $k((x_1, x_2), (y_1, y_2)) = k_1(x_1, y_1) + k_2(x_2, y_2)$
