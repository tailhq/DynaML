!!! summary ""
    Version 1.5 of DynaML, released August 11, 2017. Updates to global
    optimization api, improvements and new features in the gaussian
    process and stochastic process api.
   

## Additions

**Package** `dynaml.algebra`

 - Added support for [dual numbers](https://en.wikipedia.org/wiki/Dual_number).

   ```scala
   //Zero Dual
   val zero = DualNumber.zero[Double]  
   
   val dnum = DualNumber(1.5, -1.0) 
   val dnum1 = DualNumber(-1.5, 1.0) 
   
   //Algebraic operations: multiplication and addition/subtraction
   dnum1*dnum2
   dnum1 - dnum
   dnum*zero 
   ```

**Package** `dynaml.probability`

 - Added support for mixture distributions and mixture random variables. `MixtureRV`, `ContinuousDistrMixture` for random variables and `MixtureDistribution` for constructing mixtures of breeze distributions.

**Package** `dynaml.optimization`

 - Added `ModelTuner[T, T1]` trait as a super trait to `GlobalOptimizer[T]`
 - `GridSearch` and `CoupledSimulatedAnnealing` now extend `AbstractGridSearch` and `AbstractCSA` respectively.
 - Added `ProbGPMixtureMachine`: constructs a mixture model after a CSA or grid search routine by calculating the mixture probabilities of members of the final hyper-parameter ensemble.

### Stochastic Mixture Models

**Package** `dynaml.models`

 - Added `StochasticProcessMixtureModel`as top level class for stochastic mixture models.
 - Added `GaussianProcessMixture`: implementation of gaussian process
   mixture models.
 - Added `MVTMixture`: implementation of mixture model over
   multioutput matrix T processes.

### Kulback-Leibler Divergence

**Package** `dynaml.probability`
  
  - Added method `KL()` to `probability` package object, to calculate
    the Kulback Leibler divergence between two continuous random
    variables backed by breeze distributions. 
	
	

### Adaptive Metropolis Algorithms.

 - [AdaptiveHyperParameterMCMC](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.probability.mcmc.AdaptiveHyperParameterMCMC) which
   adapts the exploration covariance with each sample.
   
 - [HyperParameterSCAM](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.probability.mcmc.HyperParameterSCAM) adapts
   the exploration covariance for each hyper-parameter independently.


### Splines and B-Spline Generators

**Package** `dynaml.analysis`

 - [B-Spline generators](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.analysis.BSplineGenerator)
 - [Bernstein](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.analysis.BernsteinSplineGenerator$) and [Cardinal](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.analysis.CardinalBSplineGenerator$) b-spline generators.
 - Arbitrary spline functions can be created using the [`SplineGenerator`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.analysis.SplineGenerator) class.

### Cubic Spline Interpolation Kernels

**Package** `dynaml.kernels`

 - Added cubic spline interpolation kernel [`CubicSplineKernel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.kernels.CubicSplineKernel) and its ARD analogue [`CubicSplineARDKernel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.kernels.CubicSplineARDKernel) 

### Gaussian Process Models for Linear Partial Differential Equations

Based on a legacy ICML 2003 paper by [Graepel](https://www.aaai.org/Papers/ICML/2003/ICML03-033.pdf). DynaML now ships with capability of performing PDE forward and inverse inference using the Gaussian Process API.

**Package** `dynaml.models.gp`

 - [`GPOperatorModel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.models.gp.GPOperatorModel): models a quantity of interest which is governed by a linear PDE in space and time.

**Package** `dynaml.kernels`

 - [`LinearPDEKernel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.kernels.LinearPDEKernel): The core kernel primitive accepted by the `GPOperatorModel` class.

 - [`GenExpSpaceTimeKernel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.GenExpSpaceTimeKernel): a kernel of the exponential family which can serve as a handy base kernel for `LinearPDEKernel` class.


### Basis Function Gaussian Processes

DynaML now supports GP models with explicitly incorporated basis
functions as linear mean/trend functions.

**Package** `dynaml.models.gp`
 
 - [`GPBasisFuncRegressionModel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.models.gp.GPBasisFuncRegressionModel) can
   be used to create GP models with trends incorporated as a linear
   combination of basis functions.
   
### Log Gaussian Processes

 - [LogGaussianProcessModel](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5/dynaml-core/#io.github.mandar2812.dynaml.models.gp.LogGaussianProcessModel) represents
   a stochastic process whose natural logarithm follows a gaussian process.


## Improvements

**Package** `dynaml.probability`

  - Changes to `RandomVarWithDistr`: made type parameter `Dist` covariant.
  - Reform to `IIDRandomVar` hierarchy.
  
**Package** `dynaml.probability.mcmc`

 - Bug-fixes to the `HyperParameterMCMC` class. 

**General**

 - DynaML now ships with Ammonite `v1.0.0`.


## Fixes

**Package** `dynaml.optimization`

 - Corrected energy calculation in `CoupledSimulatedAnnealing`; added
   log likelihood due to hyper-prior. 
   
**Package** `dynaml.optimization`

 - Corrected energy calculation in `CoupledSimulatedAnnealing`; added
   log likelihood due to hyper-prior. 
   
   
 


