!!! summary
    "If you're not sure whether DynaML fits your requirements, this list provides a semi-comprehensive overview of available features."

## Models

Model Family | Supported | Notes
--------|-----------|-----------
[Generalized Linear Models](/core/core_glm.md) | Yes | Supports regularized least squares based models for regression as well as logistic and probit models for classification.
[Generalized Least Squares Models](/core/core_gls.md) | Yes | -
[Least Squares Support Vector Machines](/core/core_lssvm.md) | Yes | Contains implementation of dual LS-SVM applied to classification and regression.
[Gaussian Processes](/core/core_gp.md) | Yes | Supports gaussian process inference models for regression and binary classification; the binary classification GP implementation uses the Laplace approximation for posterior mode computation. For regression problems, there are also multi-output and multi-task GP implementations.
[Student T Processes](/core/core_stp.md) | Yes | Supports student T process inference models for regression, there are also multi-output and multi-task STP implementations.
[Multi-output Matrix T Process](/core/core_multi_output_t.md) | Yes | _
[Skew Gaussian Processes](/core/core_esgp.md) | Yes | Supports extended skew gaussian process inference models for regression.
[Feed forward Neural Networks](/core/core_ffn_new.md)| Yes | Can build and learn feedforward neural nets of various sizes.
[Committee/Meta Models](/core/core_model_hierarchy.md#meta-modelsmodel-ensembles) | Yes | Supports creation of gating networks or committee models.

## Optimizers & Solvers

### Parametric Solvers

Solver | Supported | Notes
--------|-----------|-----------
[Regularized Least Squares](/core/core_opt_convex.md#regularized-least-squares) | Yes | Solves the [_Tikhonov Regularization_](https://en.wikipedia.org/wiki/Tikhonov_regularization) problem exactly (not suitable for large data sets)
[Gradient Descent](/core/core_opt_convex.md#gradient-descent) | Yes | Stochastic and batch gradient descent is implemented.
[Quasi-Newton BFGS](/core/core_opt_convex.md#quasi-newton-bfgs) | Yes | Second order convex optimization (using Hessian).
[Conjugate Gradient](/core/core_opt_convex.md#conjugate-gradient) | Yes | Supports solving of linear systems of type $A.x = b$ where $A$ is a symmetric positive definite matrix.
[Committee Model Solver](/core/core_opt_convex.md#committee-model-solver) | Yes | Solves any committee based model to calculate member model coefficients or confidences.
[Back-propagation](/core/core_opt_convex.md#backpropagation-with-momentum) | Yes | Least squares based back-propagation with momentum and regularization.

### Global Optimization Solvers

Solver | Supported | Notes
--------|-----------|-----------
[Grid Search](/core/core_opt_global.md#grid-search) | Yes | Simple search over a grid of configurations.
[Coupled Simulated Annealing](/core/core_opt_global.md#coupled-simulated-annealing) | Yes | Supports vanilla (simulated annealing) along with variants of CSA such as CSA with variance (temperature) control.
[ML-II](/core/core_opt_global.md#maximum-likelihood-ml-ii)| Yes | Gradient based optimization of log marginal likelihood in Gaussian Process regression models.
