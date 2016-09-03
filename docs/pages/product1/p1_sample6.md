---
title: Workflows on Models
keywords: sample
summary: "Some workflows/pipes can be used to carry out certain operations on models such as training and hyper-parameter optimization."
sidebar: product1_sidebar
permalink: p1_sample6.html
tags: [pipes, workflow]
folder: product1
---

## Operations on Models

### Train a parametric model

```scala
trainParametricModel[
  G, T, Q, R,
  S, M <: ParameterizedLearner[G, T, Q, R, S]](
  regParameter: Double, step: Double, maxIt: Int, mini: Double)
```

* _Type_: ```DataPipe[M, M] ```
* _Result_: Takes as input a parametric model i.e. a subclass of ```ParameterizedLearner[G, T, Q, R, S]```, trains it and outputs the trained model.

### Tune a model using global optimization

```scala
modelTuning[M <: GloballyOptWithGrad](
  startingState: Map[String, Double], globalOpt: String,
  grid: Int, step: Double)
```

* _Type_: ```DataPipe[(S, S), (D, D)] ```
* _Result_: Takes as input a parametric model i.e. a subclass of ```GloballyOptimizableWithGrad```, tunes it using a global optimization procedure ```globalOpt``` and outputs the tuned model.


{% include links.html %}
