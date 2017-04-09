---
title: "Global Optimization & Hyper-parameter Selection"
sidebar: coreapi_sidebar
permalink: core_opt_global.html
folder: coreapi
---

## Model Selection Routines

These routines are also known as _global optimizers_, paradigms/algorithms such as genetic algorithms, gibbs sampling, simulated annealing, evolutionary optimization fall under this category. They can be used in situations when the objective function in not "smooth".

In DynaML they are most prominently used in hyper-parameter optimization in kernel based learning methods. All _global optimizers_ in DynaML extend the [```GlobalOptimizer```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GlobalOptimizer) trait, which implies that they provide an implementation for its ```optimize``` method.

In order to use a global optimization routine on an model, the model implementation in question must be extending the [```GloballyOptimizable```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GloballyOptimizable) trait in the [```dynaml.optimization```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.package) package, this trait has only one method called ```energy``` which is to be implemented by all sub-classes/traits.

The ```energy``` method calculates the value of the global objective function for a particular configuration i.e. for particular values of model hyper-parameters. This objective function can be defined differently for each model class (marginal likelihood for Gaussian Processes, cross validation score for parametric models, etc).

The following model selection routines are available in DynaML so far.

### Grid Search

The most elementary (naive) method of model selection is to evaluate its performance (value returned by ```energy```) on a fixed set of grid points which are initialized for the model hyper-parameters.

```scala
val kernel = ...
val noise = ...
val data = ...
val model = new GPRegression(kernel, noise, data)

val grid = 5
val step = 0.2

val gs = new GridSearch[model.type](model)
	.setGridSize(grid)
	.setStepSize(step)
	.setLogScale(false)

val startConf = kernel.state ++ noise.state
val (_, conf) = gs.optimize(startConf, opt)

model.setState(conf)
```

### Coupled Simulated Annealing

[Coupled Simulated Annealing](ftp://ftp.esat.kuleuven.be/sista/sdesouza/papers/CSA2009accepted.pdf) (CSA) is an iterative search procedure which evaluates model performance on a grid and in each iteration perturbs the grid points in a randomized manner. Each perturbed point is accepted using a certain acceptance probability which is a function of the performance on the whole grid.

Coupled Simulated Annealing can be seen as an extension to the classical Simulated Annealing algorithm, since the acceptance probability and perturbation function are design choices, we can formulate a number of variants of CSA. Any CSA-like algorithm must have the following components.


* An ensemble or grid of points $$x_i \in \Theta$$.
* A perturbation distribution or function $$P: x_i \rightarrow y_i $$.
* A coupling term $$\gamma$$ for an ensemble.
* An acceptance probability function $$A_{\Theta}(\gamma, x_i \rightarrow y_i)$$.
* An _annealing schedule_ $$T_{k}^{ac}, k = 0, 1, \cdots $$.

<br/>
The ```CoupledSimulatedAnnealing``` class has a companion [object]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.CoupledSimulatedAnnealing$) with the following available variants.
<br/>

Method | Variant |Acceptance Probability | Coupling term $$\gamma$$
--------|-----------|-----------
```SA``` | Classical _Simulated Annealing_| $$1/(1 + exp(\frac{E(y) - E(x)}{T^{ac}_{k}}))$$ | -
```MuSA``` |_Multi-state Simulated Annealing_: Direct generalization of _Simulated Annealing_| $$exp(-E(y_i))/(exp(-E(y_i)) + \gamma)$$ | $$\sum_{x_j \in \Theta}{exp(-E(x_j)/T^{ac}_{k})}$$
```BA``` | _Blind Acceptance_ CSA| $$1 - exp(-E(x_i)/T_{k}^{ac})/\gamma $$ | $$\sum_{x_j \in \Theta}{exp(-E(x_j)/T^{ac}_{k})}$$
```M``` | Modified CSA |  $$exp(E(x_i)/T_{k}^{ac})/\gamma $$ | $$\sum_{x_j \in \Theta}{exp(E(x_j)/T^{ac}_{k})}$$
```MwVC``` | Modified CSA with Variance Control: Employs an _annealing schedule_ that controls the variance of the acceptance probabilities of states | $$exp(E(x_i)/T_{k}^{ac})/\gamma $$ | $$\sum_{x_j \in \Theta}{exp(E(x_j)/T^{ac}_{k})}$$


```scala
val kernel = ...
val noise = ...
val data = ...
val model = new GPRegression(kernel, noise, data)

//The default variant of CSA is Mw-VC
val gs = new CoupledSimulatedAnnealing[model.type](model)
	.setGridSize(grid)
	.setStepSize(step)
	.setLogScale(false)
	.setVariant(CoupledSimulatedAnnealing.MuSA)

val startConf = kernel.state ++ noise.state
val (_, conf) = gs.optimize(startConf, opt)

model.setState(conf)
```

## Gradient based Model Selection

Gradient based model selection can be used if the model fitness function implemented in the ```energy``` method has differentiability properties (e.g. using marginal likelihood in the case of stochastic process inference). The [```GloballyOptWithGrad```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GloballyOptWithGrad) trait is an extension of ```GlobalOptimizer``` and adds a method ```gradEnergy``` that should return the gradient of the fitness function in each hyper-parameter in the form of a ```Map[String, Double]```.

### Maximum Likelihood ML-II

In the _Maximum Likelihood_ (ML-II) algorithm (refer to [Ramussen & Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en ) for more details), we aim to maximize the log marginal likelihood by calculating its gradient with respect to the hyper-parameters $$\theta_j$$ in each iteration and performing _steepest ascent_. The calculations are summarized below.



$$
\begin{equation}
log p(\mathbf{y}| X, \mathbf{\theta}) = - \frac{1}{2} \mathbf{y}^T K^{-1} \mathbf{y} - \frac{1}{2} log |K| - \frac{n}{2} log 2\pi
\end{equation}
$$

$$
\begin{align}
& \frac{\partial }{\partial \theta_j} log p(\mathbf{y}| X, \mathbf{\theta}) = \frac{1}{2} tr ((\mathbf{\alpha} \mathbf{\alpha}^T - K^{-1}) \frac{\partial K}{\partial \theta_j}) \\
& \mathbf{\alpha} = K^{-1} \mathbf{y}
\end{align}

$$

The [```GPMLOptimizer[I, T, M]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GPMLOptimizer) class implements ML-II, by using the ```gradEnergy``` method implemented by the ```system: M``` member value (which refers to a model extending  [```GloballyOptWithGrad```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.GloballyOptWithGrad)).

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
