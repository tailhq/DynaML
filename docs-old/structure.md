## Motivation

DynaML was born out of the need to have a performant, extensible and easy to use Machine Learning research environment. Scala was a natural choice for these requirements due to its sprawling data science ecosystem (i.e. [Apache Spark](http://spark.apache.org/)), its functional object-oriented duality and its interoperability with the Java Virtual Machine.

The DynaML distribution is divided into four principal modules.

## Modules

### Core

The heart of the DynaML distribution is the `dynaml-core` module.

The [core](/core/core_model_hierarchy.md) api consists of :

  - Model implementations
    * Parametric Models
    * Stochastic Process Models
  - Optimization solvers
  - Probability distributions/random variables
  - Kernel functions for Non parametric models


### Data workflows & Pipes

The `dynaml-pipes` module provides an API for creating modular data processing workflows.

The [pipes](/pipes/pipes.md) module aims to separate model pre-processing tasks such as cleaning data files, replacing missing or corrupt records, applying transformations on data etc.

 - Ability to create arbitrary workflows from scala functions and join them
 - Feature transformations such as wavelet transform, gaussian scaling, auto-encoders etc

### DynaML REPL

The _read evaluate print loop_ (REPL) gives the user the ability to experiment with the data pre-processing and model building process in a mix and match fashion. The DynaML shell is based on the [Ammonite](http://www.lihaoyi.com/Ammonite/) project which is an augmented Scala REPL, all the features of the Ammonite REPL are a part of the DynaML REPL.

### DynaML Examples

The module ```dynaml-examples``` contains programs which build regression and classification models on various data sets. These examples serve as case studies as well as instructional material to show the capabilities of DynaML in a hands on manner. Click [here](/repl-examples/p2_examples.md) to get started with the examples.

## Libraries Used

DynaML leverages a number of open source projects and builds on their useful features.

* [Breeze](https://github.com/scalanlp/breeze) for linear algebra operations with vectors, matrices etc.
* [Gremlin](https://github.com/tinkerpop/gremlin) for building graphs in Neural network based models.
* [Spire](https://github.com/non/spire) for creating algebraic entities like Fields, Groups etc.
* [Ammonite](http://www.lihaoyi.com/Ammonite/) for the shell environment.
* DynaML uses the newly minted [Wisp](https://github.com/quantifind/wisp) plotting library to generate aesthetic charts of common model validation metrics. There is also support for the  [JZY3D](http://jzy3d.org) scientific plotting library.

![plots](images/plot-screen.png)
