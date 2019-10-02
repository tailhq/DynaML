---
title: Workflows on Models
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
