---
layout: page
title: DynaML Shell Examples
---

## DynaML REPL


The DynaML scala shell is the first point of contact when experimenting with data analysis workflows and modeling algorithms. In this section we consider two representative examples of binary classification and regression. After starting the DynaML REPL one can start experimenting with building models for data sets. In the `data/` directory you can find a set of packaged data sets. 

------

### Regression: Boston Housing Data Set

The [boston housing](https://archive.ics.uci.edu/ml/datasets/Housing) data set is a popular multivariate regression set encountered in machine learning research. In this section we train demonstrate how to train a *Gaussian Process* regression model on this data. For a detailed introduction on *Gaussian Processes* you can refer to the book written by [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en).

To run the boston housing example run the following at the DynaML repl

```scala
TestGPHousing(kernel = new ..., noise = new ..., grid = 10,
step = 0.03, globalOpt = "GS", trainFraction = 0.45)
```

In this example we make use of the *pipes* module of DynaML which enables us to plug together arbitrary number of data processing operations. This allows us to separate data pre-processing actions from the actual model building.

{% gist mandar2812/bc5ff898ca921f22b5ee %}

------

### System Identification: Santa Fe Laser

The Santa Fe laser data is a standard benchmark data set in system identification. It serves as good starting point to start exploring time series models. It records only one observable (laser intensity), has little noise and is generated from a known physics dynamical process. Run the example at the DynaML repl prompt as follows.

```scala
SantaFeLaser(new RBFKernel(2.5), new DiracKernel(1.0),
opt = Map("globalOpt" -> "GS", "grid" -> "10", "step" -> "0.1"),
num_training = 200, num_test = 500, deltaT = 5)
```

{% gist mandar2812/0ac7ea02b73548c2e61d %}

------

### Regression: Delve Data

The [Delve](http://www.cs.toronto.edu/~delve/data/datasets.html) archive contains many data sets for model comparison and testing. A synthetic regression data set from the archive is bundled in DynaML's `data/` directory.


```scala
TestNNDelve(hidden = 2, nCounts = List(3,3),
acts = List("logsig", "logsig", "linear"),
training = 100, test = 1000,
columns = List(10,0,1,2,3,4,5,6,7,8,9),
stepSize = 0.01, maxIt = 30, mini = 1.0,
alpha = 0.5, regularization = 0.5)
```


{% gist mandar2812/4f067223d4ce7f2fba11 %}

------

### Binary Classification: Wine Quality

The [_wine quality_](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) data is commonly used to benchmark classification models. In this example program, we transform the output category (which takes values from `0` to `10`) into a binary class value such that it takes the value `1` for quality scores greater than `6` and `-1` or `0` (depending on model employed) for scores lesser than or equal to `6`. 


#### Using Neural Networks

```scala
TestNNWineQuality(hidden = 1, nCounts = List(2),
acts = List("linear", "logsig"), stepSize = 0.05, maxIt = 80,
mini = 1.0, alpha = 0.55,
training = 1000, test = 600,
regularization = 0.001,
wineType = "red")
```

{% gist mandar2812/f918bc0b52ec1b08e5bfe988a5657f9a %}


#### Using Logistic Regression

```scala
TestLogisticWineQuality(stepSize = 0.2, maxIt = 120, 
mini = 1.0, training = 800,
test = 800, regularization = 0.2, 
wineType = "red")
```

{% gist mandar2812/b309d5c26b5aba9c84415d2f7cd6d913 %}
