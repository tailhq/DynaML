---
layout: default
title: Home
noToc: true
---

[![Join the chat at https://gitter.im/mandar2812/DynaML](https://badges.gitter.im/mandar2812/DynaML.svg)](https://gitter.im/mandar2812/DynaML?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)[![Build Status](https://travis-ci.org/mandar2812/DynaML.svg?branch=branch-1.0)](https://travis-ci.org/mandar2812/DynaML)

What is DynaML?
=================
DynaML is a Scala environment for conducting research and education in Machine Learning. DynaML comes packaged with a powerful library of classes for various predictive models and a Scala REPL where one can not only build custom models but also play around with data work-flows.

![dynaml](https://cloud.githubusercontent.com/assets/1389553/13198526/4481d1b6-d80b-11e5-911b-4ba0a3e0c73e.png)

DynaML uses the newly minted [Wisp](https://github.com/quantifind/wisp) plotting library to generate aesthetic charts of common model validation metrics.

![plots](https://cloud.githubusercontent.com/assets/1389553/13198527/44834078-d80b-11e5-86bb-e16edf32d913.png)

Motivation behind DynaML
=================

DynaML was born out of the need to have a performant, extensible and easy to use Machine Learning research environment. Scala was a natural choice for these requirements due to its sprawling data science ecosystem (i.e. [Apache Spark](http://spark.apache.org/)), its functional object-oriented duality and its interoperability with the Java Virtual Machine.

Current status/Road ahead
=================

DynaML is a fledgling open source project that is in a phase of rapid expansion. Currently it supports.

* Regression with kernel based [Dual LS-SVM](/DynaML/models/#least-squares-support-vector-machines)
* Regression with [Gaussian Processes](/DynaML/models/#gaussian-processes)
* Feed forward [Neural Networks](/DynaML/models/#feed-forward-neural-networks)
* Committee Models
  - Neural Committee Models
  - Gaussian Process Committee Models
* Model Learning and Optimization
  - Gradient Descent
  - [Conjugate Gradient](/DynaML/optimization-primitives/#conjugate-gradient)
  - [Committee Model Solver](/DynaML/optimization-primitives/#committee-model-solver)
  - [Back propogation with momentum](/DynaML/optimization-primitives/#backpropagation-with-momentum)
  - [LSSVM linear solver](/DynaML/optimization-primitives/#dual-lssvm-solver)
* Model tuning
  * [Grid Search](/DynaML/optimization-primitives/#grid-search)
  * [Maximum Likelihood (ML-II)](/DynaML/optimization-primitives/#maximum-likelihood-ml-ii)
  * [Coupled Simulated Annealing](/DynaML/optimization-primitives/#coupled-simulated-annealing)
* Model validation metrics (RMSE, Area under ROC)
* Entropy based data subset selection
* Data Pipes for configurable workflows

Going ahead we aim to introduce (but not limit to)

* Sampling based Bayesian models
* Large scale committee models ([Apache Spark](http://spark.apache.org/) RDD based implementations)
* GPU support



Documentation
=============
You can refer to the project [wiki](https://github.com/mandar2812/DynaML/wiki) or [API](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#package) docs. Bear in mind that this is still at its infancy and there will be many more improvements/tweaks in the future.
