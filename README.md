# Bayes Learn

[![Build Status](https://travis-ci.org/mandar2812/bayeslearn.svg?branch=master)](https://travis-ci.org/mandar2812/bayeslearn)

![Logo](https://lh6.googleusercontent.com/8sS5eh3BbCmx5zjIT91OKWDl6eTtlxIDU1s6r-bPypvBUGPHgXICsQa_zKUw_7pj6dAGCQ62HPk9gec=w1293-h561)

Scala Library/REPL for working with Bayesian graphical models.

Introduction
============

Bayes learn is a scala library/repl for implementing and working with Probabilistic Graphical Models. The aim is to build a robust set of abstract classes and interfaces, so general graph based ML algorithms can be realized using the API.

A good introduction to Probabilistic Graphical Models can be found [here](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/131214.pdf) in [David Barber's](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage) text book. The Gaussian model implemented so far corresponds to the one discussed in Chapter 18 of the book.

Installation
============
Prerequisites: Maven to build the executables.

* Clone this repository
* Run the following.
```shell
  mvn clean compile
  mvn package
```

* Make sure you give execution permission to `bayeslearn-repl` in the `target/bin` directory.
```shell
  chmod +x target/bin/bayesLearn-repl
  target/bin/bayesLearn-repl
```
  You should get the following prompt.
  
```
       ___           ___           ___           ___           ___              
     /\  \         /\  \         |\__\         /\  \         /\  \             
    /::\  \       /::\  \        |:|  |       /::\  \       /::\  \            
   /:/\:\  \     /:/\:\  \       |:|  |      /:/\:\  \     /:/\ \  \           
  /::\~\:\__\   /::\~\:\  \      |:|__|__   /::\~\:\  \   _\:\~\ \  \          
 /:/\:\ \:|__| /:/\:\ \:\__\     /::::\__\ /:/\:\ \:\__\ /\ \:\ \ \__\         
 \:\~\:\/:/  / \/__\:\/:/  /    /:/~~/~    \:\~\:\ \/__/ \:\ \:\ \/__/         
  \:\ \::/  /       \::/  /    /:/  /       \:\ \:\__\    \:\ \:\__\           
   \:\/:/  /        /:/  /     \/__/         \:\ \/__/     \:\/:/  /           
    \::/__/        /:/  /                     \:\__\        \::/  /            
     ~~            \/__/                       \/__/         \/__/             
      ___       ___           ___           ___           ___                  
     /\__\     /\  \         /\  \         /\  \         /\__\                 
    /:/  /    /::\  \       /::\  \       /::\  \       /::|  |                
   /:/  /    /:/\:\  \     /:/\:\  \     /:/\:\  \     /:|:|  |                
  /:/  /    /::\~\:\  \   /::\~\:\  \   /::\~\:\  \   /:/|:|  |__              
 /:/__/    /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/ |:| /\__\             
 \:\  \    \:\~\:\ \/__/ \/__\:\/:/  / \/_|::\/:/  / \/__|:|/:/  /             
  \:\  \    \:\ \:\__\        \::/  /     |:|::/  /      |:/:/  /              
   \:\  \    \:\ \/__/        /:/  /      |:|\/__/       |::/  /               
    \:\__\    \:\__\         /:/  /       |:|  |         /:/  /                
     \/__/     \/__/         \/__/         \|__|         \/__/                 

Welcome to Bayes Learn v 1.0
Interactive Scala shell
STADIUS ESAT KU Leuven (2015)

bayeslearn>
  
```

Getting Started
===============

The `data/` directory contains a few sample data sets, we will be using them in the following example.

* First we create a linear Gaussian Bayesian model using a csv data set. We will assume that the last column in each line of the file is the target variable, and we build a Bayesian regression model with additive Gaussian noise.

```scala
	val config = Map("file" -> "data/ionosphere.csv", "delim" -> ",", "head" -> "false", "task" -> "classification")
	val model = GaussianLinearModel(config)
```

* We can now (optionally) add a Kernel on the model to create a generalized linear Bayesian model.

```scala
  val rbf = new RBFKernel(1.025)
  model.applyKernel(rbf)
```

```
15/04/08 16:18:27 INFO GaussianLinearModel: Calculating sample variance of the data set
Apr 08, 2015 4:18:27 PM com.github.fommil.netlib.BLAS <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
Apr 08, 2015 4:18:27 PM com.github.fommil.netlib.BLAS <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
15/04/08 16:18:27 INFO GaussianLinearModel: Using Silvermans rule of thumb to set bandwidth of density kernel
15/04/08 16:18:27 INFO GaussianLinearModel: Std Deviation of the data: DenseVector(0.6618767091095631, 0.2153531017806649, 0.6301271799415376, 0.24541745830740389, 0.5431775259758828, 0.2946765619840614, 0.5211964641055545, 0.2761229821927304, 0.5577752837852287, 0.2845775306619974, 0.5479218770489251, 0.2645605240090495, 0.5548377593832784, 0.22456828710772916, 0.5447957151077759, 0.2610149610216883, 0.5376472009654502, 0.28306794461665225, 0.49930806466116906, 0.2771708644986171, 0.5102925100414256, 0.2926837740168661, 0.5044149749875603, 0.28672626576428256, 0.5628969922087946, 0.3188982068909329, 0.49719313771932133, 0.2858619900718872, 0.45806936986691993, 0.29055696674876463, 0.39475699812037596, 0.23682328160611926)
15/04/08 16:18:27 INFO GaussianLinearModel: norm: 2.4059713445651982
15/04/08 16:18:27 INFO GaussianLinearModel: Building low rank appriximation to kernel matrix
15/04/08 16:18:27 INFO GreedyEntropySelector$: Initializing the working set, by drawing randomly from the training set
15/04/08 16:18:27 INFO GreedyEntropySelector$: Starting iterative, entropy based greedy subset selection
15/04/08 16:18:27 INFO GreedyEntropySelector$: Returning final prototype set
15/04/08 16:18:27 INFO SVMKernel$: Constructing key-value representation of kernel matrix.
15/04/08 16:18:27 INFO SVMKernel$: Dimension: 17 x 17
15/04/08 16:18:27 INFO SVMKernelMatrix: Eigenvalue decomposition of the kernel matrix using JBlas.
Apr 08, 2015 4:18:27 PM com.github.fommil.netlib.LAPACK <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
Apr 08, 2015 4:18:27 PM com.github.fommil.netlib.LAPACK <clinit>
WARNING: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
15/04/08 16:18:27 INFO SVMKernelMatrix: Eigenvalue stats: 0.020963349640349942 =< lambda =< 4.948104887300834
15/04/08 16:18:27 INFO GaussianLinearModel: Applying Feature map to data set
15/04/08 16:18:28 INFO GaussianLinearModel: DONE: Applying Feature map to data set
```

* Now we can use Gradient Descent to learn the parameters w of the Bayesian model, with priors corresponding to zero mean and variance depending on the regularization parameter.

```scala
  model.setRegParam(0.001).setMaxIteartions(100).setLearningRate(0.001).setBatchFraction(1.0).learn
```

* We can now predict the value of the targer variable given a new point consisting of a Vector of features using `model.predict()`.

* Evaluating models is easy in BayesLearn. You can create an evaluation object as follows. 

```scala
	val configtest = Map("file" -> "data/ionosphereTest.csv", "delim" -> ",", "head" -> "false")
	val met = model.evaluate(configtest)
	met.print
```

* The object `met` has a `print()` method which will dump some performance metrics in the shell. But you can also generate plots by using the `generatePlots()` method.
```
15/04/08 16:19:37 INFO BinaryClassificationMetrics: Classification Model Performance
15/04/08 16:19:37 INFO BinaryClassificationMetrics: ============================
15/04/08 16:19:37 INFO BinaryClassificationMetrics: Area under PR: 0.3313609467455621
15/04/08 16:19:37 INFO BinaryClassificationMetrics: Area under ROC: 0.33136094674556216
```

```scala
met.generatePlots
```

![Image of Plots](http://drive.google.com/uc?export=view&id=0BwmVAhMMfhhgSXV2WDNLRl9OSkE)

* Kernel based models are highly sensitive to the hyperparameters so use `model.tuneRBFKernel` to find the best value of the kernel parameters. BayesLearn will carry out a grid search over various values of the hyperparameter and use 10-fold cross-validation to find an error estimates for each value of the hyperparameter chosen. 

Documentation
=============
You can refer to the project [home page](http://mandar2812.github.io/bayeslearn/) or the [documentation](http://mandar2812.github.io/bayeslearn/target/site/scaladocs/index.html#package) for getting started with Bayes Learn. Bear in mind that this is still at its infancy and there will be many more improvements/tweaks in the future.
