---
layout: page
title: Optimization Primitives
---

## Model Solvers

Model solvers are implementations which either solve for the parameters/coefficients which determine the prediction of a model. Below is a list of all model solvers currently implemented, they are all sub-classes/subtraits of the top level optimization API. Refer to the [wiki page](https://github.com/mandar2812/DynaML/wiki/Optimization-%26-Model-Selection) on optimizers for more details on extending the API and writing your own optimizers.

### Gradient Descent

The bread and butter of any machine learning framework, the ```GradientDescent``` class in the ```io.github.mandar2812.dynaml.optimization``` package provides gradient based optimization primitives for solving optimization problems of the form.

$$
\begin{equation}
    f(w) := 
    \lambda\, R(w) +
    \frac1n \sum_{k=1}^n L(w;x_k,y_k) 
    \label{eq:regPrimal}
    \ .
\end{equation}
$$

#### Gradients


Name | Class | Equation 
------------ | ------------- | -------------
Logistic Gradient | ```LogisticGradient``` | $$ L = \frac1n \sum_{k=1}^n \log(1+\exp( -y_k w^T x_k)), y_k \in \{-1, +1\}$$
Least Squares Gradient | ```LeastSquaresGradient``` | $$ L = \frac1n \sum_{k=1}^n \|w^{T} \cdot x_k - y_k\|^2 $$



#### Updaters

Name | Class | Equation 
------------ | ------------- | -------------
$$ L_1 $$ Updater | ```L1Updater```| $$R = \|\|w\|\|_1 $$
$$ L_2 $$ Updater | ```SquaredL2Updater``` | $$R = \frac{1}{2} \|\|w\|\|^2 $$


```scala
val data: Stream[(DenseVector[Double], Double)] = ...
val num_points = data.length
val initial_params: DenseVector[Double] = ...
val optimizer = new GradientDescent(
	new LogisticGradient,
	new SquaredL2Updater
)
val params = optimizer.setRegParam(0.002)
.optimize(num_points, data, initial_params)
```

### Regularized Least Squares

This subroutine solves the regularized least squares optimization problem as shown below.

$$ 
\begin{equation} 
	\min_{w} \ \mathcal{J}_{P}(w) = \frac{1}{2} \gamma \ w^Tw + \frac{1}{2} \sum_{k = 1}^{N} (y_k - w^T \varphi(x_k))^2 
\end{equation} 
$$


```scala
val num_dim = ...
val designMatrix: DenseMatrix[Double] = ...
val response: DenseVector[Double] = ...

val optimizer = new RegularizedLSSolver()


val x = optimizer.setRegParam(0.05)
	.optimize(designMatrix.nrow, 
		(designMatrix, response),
		DenseVector.ones[Double](num_dim))
```



### Backpropagation with Momentum

This is the most common learning methods for supervised training of feed forward neural networks, the edge weights are adjusted using the _generalized delta rule_.

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

The conjugate gradient method is used to solve linear systems of the form $$Ax = b$$ where $$A$$ is a symmetric positive definite matrix.

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

The LSSVM solver solves the linear program that results from the application of the _Karush, Kuhn Tucker_ conditions on the LSSVM optimization problem.

$$
\begin{equation}
\left[\begin{array}{c|c}
   0  & 1^\intercal_v   \\ \hline
   1_v & K + \gamma^{-1} \mathit{I} 
\end{array}\right] 
\left[\begin{array}{c}
   b    \\ \hline
   \alpha  
\end{array}\right] = \left[\begin{array}{c}
   0    \\ \hline
   y  
\end{array}\right]
\end{equation}
$$


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

The committee model solver aims to find the optimum values of weights applied to the predictions of a set of base models. The weights are calculated as follows.

$$
\alpha = \frac{C^{-1} \overrightarrow{1}}{\overrightarrow{1}^T C^{-1} \overrightarrow{1}}
$$

Where $$C$$ is the sample correlation matrix of errors for all combinations of the base models calculated on the training data. 

```scala
val optimizer= new CommitteeModelSolver()
//Data Structure containing for each training point the following couple
//(predictions from base models as a vector, actual target)
val predictionsTargets: Stream[(DenseVector[Double], Double)] = ...
val params = optimizer.optimize(num_points,
	predictionsTargets,
	DenseVector.ones[Double](num_of_models))
```


------

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
