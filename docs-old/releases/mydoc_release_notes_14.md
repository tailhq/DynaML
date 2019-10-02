

!!! summary ""
    Version 1.4 of DynaML, released Sept 23, 2016, implements a number of new models (multi-output GP, student T process, random variables, etc) and features (Variance control for CSA, etc).


## Models

The following inference models have been added.

### Meta Models & Ensembles

* LSSVM committees.

### Stochastic Processes

* Multi-output, multi-task _Gaussian Process_ models as reviewed in [Lawrence et. al](https://arxiv.org/abs/1106.6251).
* _Student T Processes_: single and multi output inspired from [Shah, Ghahramani et. al](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
* Performance improvement to computation of _marginal likelihood_ and _posterior predictive distribution_ in Gaussian Process models.
* Posterior predictive distribution outputted by the `#!scala AbstractGPRegression` base class is now changed to `#!scala MultGaussianRV` which is added to the `#!scala dynaml.probability` package.

## Kernels

* Added `#!scala StationaryKernel` and `#!scala LocallyStationaryKernel` classes in the kernel APIs, converted `#!scala RBFKernel`, `#!scala CauchyKernel`, `#!scala RationalQuadraticKernel` & `#!scala LaplacianKernel` to subclasses of `#!scala StationaryKernel`

* Added `#!scala MLPKernel` which implements the _maximum likelihood perceptron_ kernel as shown [here](http://gpss.cc/gpuqss16/slides/gp_gpss16_session2.pdf).

* Added _co-regionalization kernels_ which are used in [Lawrence et. al](https://arxiv.org/abs/1106.6251) to formulate kernels for vector valued functions. In this category the following co-regionalization kernels were implemented.
  - `#!scala CoRegRBFKernel`
  - `#!scala CoRegCauchyKernel`
  - `#!scala CoRegLaplaceKernel`
  - `#!scala CoRegDiracKernel`

* Improved performance when calculating kernel matrices for composite kernels.

* Added `#!scala :*` operator to kernels so that one can create separable kernels used in _co-regionalization models_.

## Optimization

* Improved performance of `#!scala CoupledSimulatedAnnealing`, enabled use of 4 variants of _Coupled Simulated Annealing_, adding the ability to set annealing schedule using so called _variance control_ scheme as outlined in [de-Souza, Suykens et. al](ftp://ftp.esat.kuleuven.be/sista/sdesouza/papers/CSA2009accepted.pdf).

## Pipes

* Added `#!scala Scaler` and `#!scala ReversibleScaler` traits to represent transformations which input and output into the same domain set, these traits are extensions of `#!scala DataPipe`.

* Added _Discrete Wavelet Transform_ based on the _Haar_ wavelet.
