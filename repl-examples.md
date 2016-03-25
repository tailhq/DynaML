---
layout: page
title: DynaML Shell Examples
---

## DynaML REPL


The DynaML scala shell is the first point of contact when experimenting with data analysis workflows and modeling algorithms. In this section we consider two representative examples of binary classification and regression.

## Model Building

After starting the DynaML REPL one can start experimenting with building models for data sets. In the `data/` directory you can find a set of packaged data sets. We will work with the file `housing.csv` for now.


### Regression: Boston Housing Data Set

The [boston housing](https://archive.ics.uci.edu/ml/datasets/Housing) data set is a popular multivariate regression set encountered in machine learning research. In this section we train demonstrate how to train a *Gaussian Process* regression model on this data. For a detailed introduction on *Gaussian Processes* you can refer to the book written by [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en).

To run the boston housing example run the following at the DynaML repl

```scala
TestGPHousing(kernel = new ..., noise = new ..., grid = 10,
step = 0.03, globalOpt = "GS", trainFraction = 0.45)
```

In this example we make use of the *pipes* module of DynaML which enables us to plug together arbitrary number of data processing operations. This allows us to separate data pre-processing actions from the actual model building.

{% gist mandar2812/bc5ff898ca921f22b5ee %}


### System Identification: Santa Fe Laser

The Santa Fe laser data is a standard benchmark data set in system identification. It serves as good starting point to start exploring time series models. It records only one observable (laser intensity), has little noise and is generated from a known physics dynamical process. Run the example at the DynaML repl prompt as follows.

```scala
SantaFeLaser(new RBFKernel(2.5), new DiracKernel(1.0),
opt = Map("globalOpt" -> "GS", "grid" -> "10", "step" -> "0.1"),
num_training = 200, num_test = 500, deltaT = 5)
```

{% mandar2812/0ac7ea02b73548c2e61d %}
