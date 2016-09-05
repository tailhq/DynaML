---
title: Model Evaluation
sidebar: coreapi_sidebar
permalink: core_model_evaluation.html
folder: coreapi
---

Model evaluation is the litmus test for knowing if your modeling effort is headed in the right direction and for comparing various alternative models (or hypothesis) attempting to explain a phenomenon. The [```evaluation```]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.package) package contains classes and traits to calculate performance metrics for DynaML models.

Classes which implement model performance calculation can extend the [```Metrics[P]```]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.Metrics) trait. The ```Metrics``` trait requires that its sub-classes implement three methods or behaviors.

* Print out the performance metrics (whatever they may be) to the screen i.e. ```print``` method.
* Return the key performance indicators in the form of a breeze ```DenseVector[Double]```, i.e. the ```kpi``` method.

## Regression Models

Regression models are generally evaluated on a few standard metrics such as _mean square error_, _mean absolute error_, _coefficient of determination_ ($$R^2$$), etc. DynaML has implementations for single output and multi-output regression models.

### Single Output

**Small Test Set**

The [```RegressionMetrics```]({{site.baseurl}}//api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.RegressionMetrics) class takes as input a scala list containing the predictions and actual outputs and calculates the following metrics.

* _Mean Absolute Error_ (mae)
* _Root Mean Square Error_ (rmse)
* _Correlation Coefficient_ ($$\rho_{y \hat{y}}$$)
* _Coefficient of Determination_ ($$R^2$$)

```scala

//Predictions computed by any model.
val predictionAndOutputs: List[(Double, Double)] = ...

val metrics = new RegressionMetrics(predictionAndOutputs, predictionAndOutputs.length)

//Print results on screen
metrics.print

```

**Large Test Set**

The [```RegressionMetricsSpark```]({{site.baseurl}}//api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.RegressionMetricsSpark) class takes as input an [_Apache Spark_ RDD](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds) containing the predictions and actual outputs and calculates the same metrics as above.

```scala

//Predictions computed by any model.
val predictionAndOutputs: RDD[(Double, Double)] = ...

val metrics = new RegressionMetricsSpark(predictionAndOutputs, predictionAndOutputs.length)

//Print results on screen
metrics.print

```

### Multiple Outputs

The [```MultiRegressionMetrics```]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics) class calculates regression performance for multi-output models.

```scala
//Predictions computed by any model.
val predictionAndOutputs: List[(DenseVector[Double], DenseVector[Double])] = ...

val metrics = new MultiRegressionMetrics(predictionAndOutputs, predictionAndOutputs.length)

//Print results on screen
metrics.print
```

## Classification Models

Currently (as of v1.4) there is only a binary classification implementation for calculating model performance.

### Binary Classification

**Small Test Sets**

The [```BinaryClassificationMetrics```]({{site.baseurl}}//api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.BinaryClassificationMetrics) class calculates the following performance indicators.

* Classification accuracy
* F-measure
* Precision-Recall Curve (and area under it).
* Receiver Operating Characteristic (and area under it)


```scala

val scoresAndLabels: List[(Double, Double)] = ...

//Set logisticFlag = true in case outputs are produced via logistic regression
val metrics = new BinaryClassificationMetrics(
          scoresAndLabels,
          scoresAndLabels.length,
          logisticFlag = true)

metrics.print
```

**Large Test Sets**

The [```BinaryClassificationMetricsSpark```]({{site.baseurl}}//api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.evaluation.BinaryClassificationMetricsSpark) class takes as input an [_Apache Spark_ RDD](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds) containing the predictions and actual labels and calculates the same metrics as above.
