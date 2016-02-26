# DynaML

[![Build Status](https://travis-ci.org/mandar2812/DynaML.svg?branch=branch-1.0)](https://travis-ci.org/mandar2812/DynaML)

Aim
============

DynaML is a scala library/repl for implementing and working with general Machine Learning models. Machine Learning/AI applications make heavy use of various entities such as graphs, vectors, matrices etc as well as classes of mathematical models which deal with broadly three kinds of tasks, prediction, classification and clustering.

The aim is to build a robust set of abstract classes and interfaces, which can be extended easily to implement advanced models for small and large scale applications.

But the library can also be used as an educational/research tool for multi scale data analysis. 

Currently DynaML supports.

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


Installation
============

Prerequisites
------------

* *nix system (linux or Mac OSX)
* Maven

Steps
-------

* Clone this repository
* Run the following.
```shell
  mvn clean compile
  mvn package
```

* Make sure you give execution permission to `DynaML` in the `target/bin` directory.
```shell
  chmod +x target/bin/DynaML
  target/bin/DynaML
```
  You should get the following prompt.
  
```
    ___       ___       ___       ___       ___       ___   
   /\  \     /\__\     /\__\     /\  \     /\__\     /\__\  
  /::\  \   |::L__L   /:| _|_   /::\  \   /::L_L_   /:/  /  
 /:/\:\__\  |:::\__\ /::|/\__\ /::\:\__\ /:/L:\__\ /:/__/   
 \:\/:/  /  /:;;/__/ \/|::/  / \/\::/  / \/_/:/  / \:\  \   
  \::/  /   \/__/      |:/  /    /:/  /    /:/  /   \:\__\  
   \/__/               \/__/     \/__/     \/__/     \/__/  

Welcome to DynaML v 1.2
Interactive Scala shell

DynaML>
```

Getting Started
===============
Refer to the DynaML [wiki](https://github.com/mandar2812/DynaML/wiki) to learn more.
