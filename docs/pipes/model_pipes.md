---
title: Model Pipes
---


## DynaML Model Pipes


We saw in the previous section that certain operations like training/tuning of models are expressed as pipes which take input the relevant model and perform an operation on it. But it is evident that the model creation itself is a common step in the data analysis workflow, therefore one needs library pipes which instantiate DynaML machine learning models given the training data and other relevant inputs. Model creation pipes are not in the `#!scala DynaMLPipe` object but exist as an independent class hierarchy. Below we explore a section of it.


### Generalized Linear Model pipe

```scala
GLMPipe[T, Source](
  pre: (Source) => Stream[(DenseVector[Double], Double)],
  map: (DenseVector[Double]) => (DenseVector[Double]) = identity _,
  task: String = "regression", modelType: String = "")
```

* _Type_: `#!scala DataPipe[Source, GeneralizedLinearModel[T]]`
* _Result_: Takes as input a data of type `#!scala Source` and outputs a _Generalized Linear Model_.

### Gaussian Process Regression Model Pipe

```scala
GPRegressionPipe[
M <: AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)], DenseVector[Double]],
Source](
  pre: (Source) => Seq[(DenseVector[Double], Double)],
  cov: LocalScalarKernel[DenseVector[Double]],
  n: LocalScalarKernel[DenseVector[Double]],
  order: Int = 0, ex: Int = 0)
```

* _Type_: `#!scala DataPipe[Source, M]`
* _Result_: Takes as input data of type `#!scala Source` and intializes a _Gaussian Process_ regression model as the output.

### Dual LS-SVM Model Pipe

```scala
DLSSVMPipe[Source](
  pre: (Source) => Stream[(DenseVector[Double], Double)],
  cov: LocalScalarKernel[DenseVector[Double]],
  task: String = "regression")
```

* _Type_: `#!scala DataPipe[Source, DLSSVM]`
* _Result_: Takes as input data of type `#!scala Source` and intializes a _LS-SVM_ regression/classification model as the output.
