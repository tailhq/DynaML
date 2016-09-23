---
title: Release notes 1.4
tags: [getting_started,release_notes]
keywords: release notes, announcements, what's new, new features
last_updated: July 16, 2016
summary: "Version 1.4 of DynaML, released Sept 23, 2016, implements a number of new models (multi-output GP, student T process, random variables, etc) and features (Variance control for CSA, etc)."
sidebar: mydoc_sidebar
permalink: mydoc_release_notes_14.html
folder: mydoc
---

## Models

The following inference models have been added.

### Meta Models & Ensembles

* LSSVM committees.

### Stochastic Processes

* Multi-output, multi-task _Gaussian Process_ models as reviewed in [Lawrence et. al](https://arxiv.org/abs/1106.6251).
* _Student T Processes_: single and multi output inspired from [Shah, Ghahramani et. al](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
* Performance improvement to computation of _marginal likelihood_ and _posterior predictive distribution_ in Gaussian Process models.
* Posterior predictive distribution outputted by the ```AbstractGPRegression``` base class is now changed to ```MultGaussianRV``` which is added to the ```dynaml.probability``` package.

## Kernels

* Added ```StationaryKernel``` and ```LocallyStationaryKernel``` classes in the kernel APIs, converted ```RBFKernel```, ```CauchyKernel```, ```RationalQuadraticKernel``` & ```LaplacianKernel``` to subclasses of ```StationaryKernel```

* Added ```MLPKernel``` which implements the _maximum likelihood perceptron_ kernel as shown [here](http://gpss.cc/gpuqss16/slides/gp_gpss16_session2.pdf).

* Added _co-regionalization kernels_ which are used in [Lawrence et. al](https://arxiv.org/abs/1106.6251) to formulate kernels for vector valued functions. In this category the following co-regionalization kernels were implemented.
  - ```CoRegRBFKernel```
  - ```CoRegCauchyKernel```
  - ```CoRegLaplaceKernel```
  - ```CoRegDiracKernel```

* Improved performance when calculating kernel matrices for composite kernels.

* Added ```:*``` operator to kernels so that one can create separable kernels used in _co-regionalization models_.

## Optimization

* Improved performance of ```CoupledSimulatedAnnealing```, enabled use of 4 variants of _Coupled Simulated Annealing_, adding the ability to set annealing schedule using so called _variance control_ scheme as outlined in [de-Souza, Suykens et. al](ftp://ftp.esat.kuleuven.be/sista/sdesouza/papers/CSA2009accepted.pdf).

## Pipes

* Added ```Scaler``` and ```ReversibleScaler``` traits to represent transformations which input and output into the same domain set, these traits are extensions of ```DataPipe```.

* Added _Discrete Wavelet Transform_ based on the _Haar_ wavelet.

{% include links.html %}
