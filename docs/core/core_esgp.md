!!! summary
    The _Extended Skew Gaussian Process_ (ESGP) uses the [MESN](/core/core_prob_dist/#mesn) distribution to define its finite dimensional probability distribution. It can be viewed as an generalization of the _Gaussian Process_ because when its skewness parameter approaches zero, the calculated probabilities are very close to gaussian probabilities.


The ESGP model uses the conditioning property of the MESN distribution, just like the multivariate normal distribution, the MESN retains its form when conditioned on a subset of its dimensions.

Creating an ESGP model is very similar to creating a GP model in DynaML. The class [`#!scala ESGPModel[T, I]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.models.sgp.ESGPModel) can be instantiated much like the [`#!scala AbstractGPRegressionModel[T, I]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.4.2/dynaml-core/#io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel), using the `apply` method.

```scala
//Obtain the data, some generic type
val trainingdata: DataType = ...

val kernel: LocalScalarKernel[I] = _
val noiseKernel: LocalScalarKernel[I] = _
val meanFunc: DataPipe[I, Double] = _

val lambda = 1.5
val tau = 0.5

//Define how the data is converted to a compatible type
implicit val transform: DataPipe[DataType, Seq[(I, Double)]] = _

val model = ESGPModel(
  kernel, noiseKernel,
  meanFunc, lambda, tau,
  trainingData)
```
