!!! summary
    Model pipes define pipelines which involve predictive models.

!!! note
    The classes described here exist in the [`dynaml.modelpipe`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.modelpipe.package) package of the [`dynaml-core`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#package) module. Although they are not strictly part of the pipes module, they are included here for clarity and continuity.  

The pipes module gives the user the ability to create workflows of arbitrary complexity. In order to enable end to end machine learning, we need pipelines which involve predictive models. These pipelines can be of two types.

 - Pipelines which take data as input and output a predictive model.

      It is evident that the model creation itself is a common step in the data analysis workflow, therefore one needs library pipes which create machine learning models given the training data and other relevant inputs.

 - Pipelines which encapsulate predictive models and generate predictions for test data splits.

      Once a model has been tuned/trained, it can be a part of a pipeline which generates predictions for previously unobserved data.


## Model Creation

All pipelines which return predictive models as outputs extend the [`#!scala ModelPipe`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.modelpipe.ModelPipe) trait.

### Generalized Linear Model Pipe

```scala
//Pre-process data
val pre: (Source) => Stream[(DenseVector[Double], Double)] = _
val feature_map: (DenseVector[Double]) => (DenseVector[Double]) = _

val glm_pipe =
  GLMPipe[(DenseMatrix[Double], DenseVector[Double]), Source](
    pre, map, task = "regression",
    modelType = "")

val dataSource: Source = _

val glm_model = glm_pipe(dataSource)
```

* _Type_: `#!scala DataPipe[Source, GeneralizedLinearModel[T]]`
* _Result_: Takes as input a data of type `#!scala Source` and outputs a [_Generalized Linear Model_](/core/core_glm.md).

### Generalized Least Squares Model Pipe

```scala
val kernel: LocalScalarKernel[DenseVector[Double]]
val gls_pipe2 = GeneralizedLeastSquaresPipe2(kernel)

val featuremap: (DenseVector[Double]) => (DenseVector[Double]) = _
val data: Stream[(DenseVector[Double], Double)] = _

val gls_model = gls_pipe2(data, featuremap)
```

* _Type_: `#!scala DataPipe2[Stream[(DenseVector[Double], Double)], DataPipe[DenseVector[Double], DenseVector[Double]], GeneralizedLeastSquaresModel]]`
* _Result_: Takes as inputs data and a feature mapping and outputs a [_Generalized Least Squares Model_](/core/core_gls.md).


### Gaussian Process Regression Model Pipe

```scala

//Pre-process data
val pre: (Source) => Stream[(DenseVector[Double], Double)] = _
//Declare kernel and noise
val kernel: LocalScalarKernel[DenseVector[Double]] = _
val noise: LocalScalarKernel[DenseVector[Double]] = _

GPRegressionPipe(
  pre, kernel, noise,
  order: Int = 0, ex: Int = 0)
```

* _Type_: `#!scala DataPipe[Source, M]`
* _Result_: Takes as input data of type `#!scala Source` and outputs a [_Gaussian Process_ regression](/core/core_gp.md) model as the output.

### Dual LS-SVM Model Pipe

```scala
//Pre-process data
val pre: (Source) => Stream[(DenseVector[Double], Double)] = _
//Declare kernel
val kernel: LocalScalarKernel[DenseVector[Double]] = _

DLSSVMPipe(pre, kernel, task = "regression")
```

* _Type_: `#!scala DataPipe[Source, DLSSVM]`
* _Result_: Takes as input data of type `#!scala Source` and outputs a [_LS-SVM_](/core/core_lssvm.md) regression/classification model as the output.


## Model Prediction

Prediction pipelines encapsulate predictive models, the [`#!scala ModelPredictionPipe`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.modelpipe.ModelPredictionPipe) class provides an expressive API for creating prediction pipelines.

```scala

//Any model
val model: Model[T, Q, R] = _

//Data pre and post processing
val preprocessing: DataPipe[P, Q] = _
val postprocessing: DataPipe[R, S] = _

val prediction_pipeline = ModelPredictionPipe(
  preprocessing,
  model,
  postprocessing)

//In case no pre or post processing is done.
val prediction_pipeline2 = ModelPredictionPipe(model)

//Incase feature and target scaling is performed

val featureScaling: ReversibleScaler[Q] = _
val targetScaling: ReversibleScaler[R] = _

val prediction_pipeline3 = ModelPredictionPipe(
  featureScaling,
  model,
  targetScaling)

```
