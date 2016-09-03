---
title: Supported features
tags:
  - getting_started
keywords: "features, capabilities, scalability"
last_updated: "August 28, 2016"
summary: "If you're not sure whether DynaML fits your requirements, this list provides a semi-comprehensive overview of available features."
published: true
sidebar: mydoc_sidebar
permalink: mydoc_supported_features.html
folder: mydoc
---

## Models

Model Family | Supported | Notes
--------|-----------|-----------
Generalized Linear Models | Yes | Supports regularized least squares based models for regression as well as logistic and probit models for classification.
Least Squares Support Vector Machines| Yes | Contains implementation of dual LS-SVM applied to classification and regression.
Gaussian Processes| Yes | Supports gaussian process inference models for regression and binary classification; the binary classification GP implementation uses the Laplace approximation for posterior mode computation. For regression problems, there are also multi-output and multi-task GP implementations.
Feed forward Neural Networks| Yes | Can build and learn feedforward neural nets of various sizes.
Committee/Meta Models| Yes | Supports creation of gating networks or committee models.

## Optimizers & Solvers

### Parametric Solvers

Solver | Supported | Notes
--------|-----------|-----------
Gradient Descent | Yes | Stochastic and batch gradient descent is implemented.
Conjugate Gradient | Yes | Supports solving of linear systems of type $$ A.x = b $$, where $$A$$ is a symmetric positive definite matrix.
Committee Model Solver | Yes | Solves any committee based model to calculate member model coefficients or confidences.
Back-propagation | Yes | Least squares based back-propagation with momentum and regularization.

### Global Optimization Solvers

Solver | Supported | Notes
--------|-----------|-----------
Grid Search | Yes | Simple search over a grid of configurations.
Coupled Simulated Annealing | Yes | Supports vanilla (simulated annealing) along with variants of CSA such as CSA with variance (temperature) control.


{% include links.html %}
