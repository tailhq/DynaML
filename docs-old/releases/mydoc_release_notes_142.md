
!!! summary ""
    Version 1.4.2 of DynaML, released May 7, 2017. Updates, improvements and new features.


## Core API

### Additions


**Package** `#!scala dynaml.models.neuralnets`

 - Added `#!scala GenericAutoEncoder[LayerP, I]`, the class `#!scala AutoEncoder` is now **deprecated**
 - Added `#!scala GenericNeuralStack[P, I, T]` as a base class for Neural Stack API
 - Added `#!scala LazyNeuralStack[P, I]` where the layers are lazily spawned.

**Package** `#!scala dynaml.kernels`

 - Added `#!scala ScaledKernel[I]` representing kernels of scaled Gaussian Processes.

**Package** `#!scala dynaml.models.bayes`

 - Added `#!scala *` method to `#!scala GaussianProcessPrior[I, M]` which creates a scaled Gaussian Process prior using the newly minted `#!scala ScaledKernel[I]` class
 - Added Kronecker product GP priors with the `#!scala CoRegGPPrior[I, J, M]` class

**Package** `#!scala dynaml.models.stp`

  - Added multi-output Students' T Regression model of [Conti & O' Hagan](http://www.sciencedirect.com/science/article/pii/S0378375809002559) in class `#!scala MVStudentsTModel`

**Package** `#!scala dynaml.probability.distributions`

   - Added `#!scala HasErrorBars[T]` generic trait representing distributions which can generate confidence intervals around their mean value.


### Improvements


**Package** `#!scala dynaml.probability`

 - Fixed issue with creation of `#!scala MeasurableFunction` instances from `#!scala RandomVariable` instances

**Package** `#!scala dynaml.probability.distributions`

  - Changed error bar calculations and sampling of Students T distributions (vector and matrix) and Matrix Normal distribution.

**Package** `#!scala dynaml.models.gp`

  - Added _Kronecker_ structure speed up to `#!scala energy` (marginal likelihood) calculation of multi-output GP models

**Package** `#!scala dynaml.kernels`
  - Improved implicit paramterization of Matern Covariance classes


**General**

  - Updated breeze version to latest.
  - Updated Ammonite version to latest
