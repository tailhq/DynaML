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
[Generalized Linear Models]({{site.baseurl}}) | Yes | Supports regularized least squares based models for regression as well as logistic and probit models for classification.
[Least Squares Support Vector Machines]({{site.baseurl}}/core_lssvm.html) | Yes | Contains implementation of dual LS-SVM applied to classification and regression.
[Gaussian Processes]({{site.baseurl}}/core_gp.html) | Yes | Supports gaussian process inference models for regression and binary classification; the binary classification GP implementation uses the Laplace approximation for posterior mode computation. For regression problems, there are also multi-output and multi-task GP implementations.
[Feed forward Neural Networks]({{site.baseurl}}//core_ann.html)| Yes | Can build and learn feedforward neural nets of various sizes.
[Committee/Meta Models]({{site.baseurl}}/core_model_hierarchy.html#meta-modelsmodel-ensembles) | Yes | Supports creation of gating networks or committee models.

## Optimizers & Solvers

### Parametric Solvers

Solver | Supported | Notes
--------|-----------|-----------
[Regularized Least Squares]({{site.baseurl}}/core_opt_convex.html#regularized-least-squares) | Yes | Solves the [_Tikhonov Regularization_](https://en.wikipedia.org/wiki/Tikhonov_regularization) problem exactly (not suitable for large data sets)
[Gradient Descent]({{site.baseurl}}/core_opt_convex.html#gradient-descent) | Yes | Stochastic and batch gradient descent is implemented.
[Quasi-Newton BFGS]({{site.baseurl}}/core_opt_convex.html#quasi-newton-bfgs) | Yes | Second order convex optimization (using Hessian).
[Conjugate Gradient]({{site.baseurl}}/core_opt_convex.html#conjugate-gradient) | Yes | Supports solving of linear systems of type $$ A.x = b $$, where $$A$$ is a symmetric positive definite matrix.
[Committee Model Solver]({{site.baseurl}}/core_opt_convex.html#committee-model-solver) | Yes | Solves any committee based model to calculate member model coefficients or confidences.
[Back-propagation]({{site.baseurl}}/core_opt_convex.html#backpropagation-with-momentum) | Yes | Least squares based back-propagation with momentum and regularization.

### Global Optimization Solvers

Solver | Supported | Notes
--------|-----------|-----------
[Grid Search]({{site.baseurl}}/core_opt_global.html#grid-search) | Yes | Simple search over a grid of configurations.
[Coupled Simulated Annealing]({{site.baseurl}}/core_opt_global.html#coupled-simulated-annealing) | Yes | Supports vanilla (simulated annealing) along with variants of CSA such as CSA with variance (temperature) control.
[ML-II]({{site.baseurl}}/core_opt_global.html#maximum-likelihood-ml-ii)| Yes | Gradient based optimization of log marginal likelihood in Gaussian Process regression models.

{% include links.html %}
