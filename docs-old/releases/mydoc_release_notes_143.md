!!! summary ""
    Version 1.4.3 of DynaML, released June 13, 2017. Updates, improvements and new features.

## DynaML REPL

### Additions


 **Module** `scripts`
 
 - Added `gp_mcmc_santafe.sc` worksheet to try new MCMC feature on GP models; applied on the Santa Fe laser data set. 
 
 **General**
 
   - Updated Ammonite version to `0.9.9`

## Pipes API

### Additions


**Package** `dynaml.pipes`
 
- Added `._1` and `._2` members in `#!scala ParallelPipe` 



## Core API

### Additions


**Package** `dynaml.models.neuralnets`

 - Added [SELU](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.models.neuralnets.VectorSELU) activation function proposed by [Hochreiter et. al](https://arxiv.org/abs/1706.02515)

**Package** `dynaml.models.bayes`

 - Added `*` method to `#!scala CoRegGPPrior` which scales it with to a `#!scala ParallelPipe`

**Package** `dynaml.probability.mcmc`

 - Added `#!scala HyperParameterMCMC` for performing MCMC sampling for models extending `#!scala GloballyOptimizable`.

**Package** `dynaml.utils`

 - Added trait `#!scala HyperParameters` outlining methods that must be implemented by entities having hyper-parameters

 - Added `#!scala MeanScaler`, `#!scala PCAScaler` to perform mean centering and PCA on data sets. Also added to `#!scala DynaMLPipe` pipe library.
 
 - Added tail recursive computation of the [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_polynomials) polynomials of the first and second kind in [`#!scala utils.chebyshev`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.utils.package). 

 
 
 
### Improvements

 
**Package** `dynaml.models.bayes`
 
   - Added `#!scala trendParamsEncoder` which converts the trend/mean parameters into a scala `#!scala Map[String, Double]` making them 
     consistent with covariance parameters. Added to `#!scala GaussianProcessPrior` and `#!scala ESGPPrior` families.
     
     

