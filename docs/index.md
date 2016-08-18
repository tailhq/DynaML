---
layout: default
title: Home
noToc: true
---

[![Join the chat at https://gitter.im/mandar2812/DynaML](https://badges.gitter.im/mandar2812/DynaML.svg)](https://gitter.im/mandar2812/DynaML?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)[![Build Status](https://travis-ci.org/mandar2812/DynaML.svg?branch=branch-1.0)](https://travis-ci.org/mandar2812/DynaML)

What is DynaML?
=================
DynaML is a Scala environment for conducting research and education in Machine Learning. DynaML comes packaged with a powerful library of classes for various predictive models and a Scala REPL where one can not only build custom models but also play around with data work-flows.

![dynaml]({{site.baseurl}}/images/screenshot.png)

DynaML uses the newly minted [Wisp](https://github.com/quantifind/wisp) plotting library to generate aesthetic charts of common model validation metrics.

![plots]({{site.baseurl}}/images/plot-screen.png)

Motivation behind DynaML
=================

DynaML was born out of the need to have a performant, extensible and easy to use Machine Learning research environment. Scala was a natural choice for these requirements due to its sprawling data science ecosystem (i.e. [Apache Spark](http://spark.apache.org/)), its functional object-oriented duality and its interoperability with the Java Virtual Machine.

Current status/Road ahead
=================

DynaML is a fledgling open source project that is in a phase of rapid expansion. Currently it supports.

* Generalized Linear Models
  - [Regularized Ordinary Least Squares]({{site.baseurl}}/models/#regularized-least-squares)
  - [Logistic and Probit Models]({{site.baseurl}}/models/#logistic--probit-regression) for binary classification
* Regression and Classification with kernel based [Dual LS-SVM]({{site.baseurl}}/models/#least-squares-support-vector-machines)
* Regression and binary classificationwith [Gaussian Processes]({{site.baseurl}}/models/#gaussian-processes)
* Feed forward [Neural Networks]({{site.baseurl}}/models/#feed-forward-neural-networks)
* Committee Models
  - [Neural Committee Models]({{site.baseurl}}/models/#neural-committee-models)
  - Gaussian Process Committee Models
* Model Learning and Optimization
  - Gradient Descent
  - [Conjugate Gradient]({{site.baseurl}}/optimization-primitives/#conjugate-gradient)
  - [Committee Model Solver]({{site.baseurl}}/optimization-primitives/#committee-model-solver)
  - [Back propogation with momentum]({{site.baseurl}}/optimization-primitives/#backpropagation-with-momentum)
  - [LSSVM linear solver]({{site.baseurl}}/optimization-primitives/#dual-lssvm-solver)
* Model tuning
  * [Grid Search]({{site.baseurl}}/optimization-primitives/#grid-search)
  * [Maximum Likelihood (ML-II)]({{site.baseurl}}/optimization-primitives/#maximum-likelihood-ml-ii)
  * [Coupled Simulated Annealing]({{site.baseurl}}/optimization-primitives/#coupled-simulated-annealing)
* Model validation metrics (RMSE, Area under ROC)
* Entropy based data subset selection
* [Data Pipes]({{site.baseurl}}/data-pipes/) for configurable workflows

Going ahead we aim to introduce (but not limit to)

* Sampling based Bayesian models
* Large scale committee models ([Apache Spark](http://spark.apache.org/) RDD based implementations)
* GPU support



Documentation
=============
You can refer to the project [wiki](https://github.com/mandar2812/DynaML/wiki) or [API](http://mandar2812.github.io/DynaML/target/site/index.html#package) docs. Bear in mind that this is still at its infancy and there will be many more improvements/tweaks in the future.
