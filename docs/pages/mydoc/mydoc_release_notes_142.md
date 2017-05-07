---
title: Release notes 1.4.2
tags: [getting_started,release_notes]
keywords: release notes, announcements, what's new, new features
last_updated: May 7, 2017
summary: "Version 1.4.2 of DynaML, released May 7, 2017. Updates, improvements and new features."
sidebar: mydoc_sidebar
permalink: mydoc_release_notes_142.html
folder: mydoc
---

## Core API

### Additions


**Package** `dynaml.models.neuralnets`

 - Added `GenericAutoEncoder[LayerP, I]`, the class `AutoEncoder` is now **deprecated** 
 - Added `GenericNeuralStack[P, I, T]` as a base class for Neural Stack API
 - Added `LazyNeuralStack[P, I]` where the layers are lazily spawned.

**Package** `dynaml.kernels`

 - Added `ScaledKernel[I]` representing kernels of scaled Gaussian Processes.

**Package** `dynaml.models.bayes`

 - Added `*` method to `GaussianProcessPrior[I, M]` which creates a scaled Gaussian Process prior using the newly minted `ScaledKernel[I]` class
 - Added Kronecker product GP priors with the `CoRegGPPrior[I, J, M]` class

**Package** `dynaml.models.stp`

  - Added multi-output Students' T Regression model of [Conti & O' Hagan](http://www.sciencedirect.com/science/article/pii/S0378375809002559) in class `MVStudentsTModel`

**Package** `dynaml.probability.distributions`

   - Added `HasErrorBars[T]` generic trait representing distributions which can generate confidence intervals around their mean value.


### Improvements


**Package** `dynaml.probability`

 - Fixed issue with creation of `MeasurableFunction` instances from `RandomVariable` instances
 
**Package** `dynaml.probability.distributions`

  - Changed error bar calculations and sampling of Students T distributions (vector and matrix) and Matrix Normal distribution.

**Package** `dynaml.models.gp`

  - Added _Kronecker_ structure speed up to `energy` (marginal likelihood) calculation of multi-output GP models

**Package** `dynaml.kernels`
  - Improved implicit paramterization of Matern Covariance classes


**General**

  - Updated breeze version to latest.
  - Updated Ammonite version to latest
