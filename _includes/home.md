
[![Build Status](https://travis-ci.org/mandar2812/DynaML.svg?branch=branch-1.0)](https://travis-ci.org/mandar2812/DynaML)

What is DynaML?
=================
DynaML is a Scala environment for conducting research and education in Machine Learning. DynaML comes packaged with a powerful library of classes for various predictive models and a Scala REPL where one can not only build custom models but also play around with data work-flows.

[[images/dynaml.png]]

DynaML uses the newly minted [Wisp](https://github.com/quantifind/wisp) plotting library to generate aesthetic charts of common model validation metrics.

[[images/plots.png]]

Motivation behind DynaML
=================

DynaML was born out of the need to have a performant, extensible and easy to use Machine Learning research environment. Scala was a natural choice for these requirements due to its sprawling data science ecosystem (i.e. [Apache Spark](http://spark.apache.org/)), its functional object-oriented duality and its interoperability with the Java Virtual Machine.

Current status/Road ahead
=================

DynaML is a fledgling open source project that is in a phase of rapid expansion. Currently it supports.

* Regression with kernel based Dual LS-SVM
* Regression with Gaussian Processes
* Feed forward Neural Networks
* Committee Models
  - Neural Committee Models
  - Gaussian Process Committee Models
* Model Learning and Optimization
  - Gradient Descent
  - Conjugate Gradient
  - Committee Model Solver
  - Back propogation with momentum
  - LSSVM linear solver
* Model tuning
  * Grid Search
  * Maximum Likelihood (ML-II)
  * Coupled Simulated Annealing
* Model validation metrics (RMSE, Area under ROC)
* Entropy based data subset selection
* Data Pipes for configurable workflows

Going ahead we aim to introduce (but not limit to)

* Sampling based Bayesian models
* Large scale committee models ([Apache Spark](http://spark.apache.org/) RDD based implementations)
* GPU support



Documentation
=============
You can refer to the project [documentation]({{site.url}}/documentation) for getting started with DynaML. Bear in mind that this is still at its infancy and there will be many more improvements/tweaks in the future.
