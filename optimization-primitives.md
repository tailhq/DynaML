---
layout: page
title: Optimization Primitives
---

## Model Solvers

Model solvers are implementations which either solve for the parameters/coefficients which determine the prediction of a model. Below is a list of all model solvers currently implemented, they are all sub-classes/subtraits of the top level optimization API. Refer to the [wiki page](https://github.com/mandar2812/DynaML/wiki/Optimization-%26-Model-Selection) on optimizers for more details.

### Backpropagation with Momentum

```scala
val data: Stream[(DenseVector[Double], DenseVector[Double])] = ...

val initParam = FFNeuralGraph(num_inputs = data.head._1.length,
	num_outputs = data.head._2.length, 
	hidden_layers = 1, List("logsig", "linear"),
	List(5))

val optimizer = new BackPropogation()
	.setNumIterations(100)
	.setStepSize(0.01)

val newparams = optimizer.optimize(data.length, data, initParam)
```

### Conjugate Gradient

```scala
val num_dim = ...
val A: DenseMatrix[Double] = ...
val b: DenseVector[Double] = ...

///Solves A.x = b
val x = ConjugateGradient.runCG(A, b,
	DenseVector.ones[Double](num_dim),
	epsilon = 0.005, MAX_ITERATIONS = 50)
```

### Dual LSSVM Solver

```scala
val data: Stream[(DenseVector[Double], Double)] = ...

val kernelMatrix: DenseMatrix[Double] = ...

val initParam =  DenseVector.ones[Double](num_points+1)

val optimizer =	new LSSVMLinearSolver()

val alpha = optimizer.optimize(num_points,
	(kernelMatrix, DenseVector(data.map(_._2).toArray)),
	initParam)
```

### Committee Model Solver

```scala
val optimizer= new CommitteeModelSolver()
//Data Structure containing for each training point the following couple
//(predictions from base models as a vector, actual target)
val predictionsTargets: Stream[(DenseVector[Double], Double)] = ...
val params = optimizer.optimize(num_points,
	predictionsTargets,
	DenseVector.ones[Double](num_of_models))
```

## Model Selection Routines

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
