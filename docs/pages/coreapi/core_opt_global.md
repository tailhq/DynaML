---
title: "Global Optimization & Hyper-parameter Selection"
sidebar: coreapi_sidebar
permalink: core_opt_global.html
folder: coreapi
---

## Model Selection Routines

These routines are also known as _global optimizers_, paradigms/algorithms such as genetic algorithms, gibbs sampling, simulated annealing, evolutionary optimization fall under this category. They can be used in situations when the objective function in not "smooth". In DynaML they are most prominently used in hyper-parameter optimization in kernel based learning methods.


### Grid Search

```scala
val kernel = ...
val noise = ...
val data = ...
val model = new GPRegression(kernel, noise, data)

val gs = new GridSearch[model.type](model)
	.setGridSize(grid)
	.setStepSize(step)
	.setLogScale(false)

val startConf = kernel.state ++ noise.state
val (_, conf) = gs.optimize(startConf, opt)

model.setState(conf)
```

### Coupled Simulated Annealing


```scala
val kernel = ...
val noise = ...
val data = ...
val model = new GPRegression(kernel, noise, data)

val gs = new CoupledSimulatedAnnealing[model.type](model)
	.setGridSize(grid)
	.setStepSize(step)
	.setLogScale(false)

val startConf = kernel.state ++ noise.state
val (_, conf) = gs.optimize(startConf, opt)

model.setState(conf)
```


### Maximum Likelihood ML-II

```scala
val kernel = ...
val noise = ...
val data = ...
val model = new GPRegression(kernel, noise, data)

val ml = new GPMLOptimizer[DenseVector[Double],
	Seq[(DenseVector[Double], Double)],
	GPRegression](model)

val startConf = kernel.state ++ noise.state
val (_, conf) = ml.optimize(startConf, opt)

model.setState(conf)
```
