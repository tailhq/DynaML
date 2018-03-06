
## Model Classes

In DynaML all model implementations fit into a well defined class hierarchy. In fact every DynaML machine learning model is an extension of the [`#!scala Model[T,Q,R]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.Model) trait.

!!! note "`#!scala Model[T,Q,R]`"
    The `#!scala Model` trait is quite bare bones: machine learning models are viewed as objects containing two parts or components.

      1. A training data set (of type `#!scala T`).  

      2. A method `#!scala predict(point: Q):R` to generate a prediction of type `#!scala R` given a data point of type `#!scala Q`.


## Parameterized Models

Many predictive models calculate predictions by formulating an expression which includes a set of parameters which are used along with the data points to generate predictions, the [```ParameterizedLearner[G, T, Q, R, S]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.ParameterizedLearner) class represents a skeleton for all parametric machine learning models such as [_Generalized Linear Models_](core_glm.md), [_Neural Networks_](core_ann.md), etc.

!!! tip
    The defining characteristic of classes which extend `#!scala ParameterizedLearner` is that they must contain a member variable ```optimizer: RegularizedOptimizer[T, Q, R, S]``` which represents a [regularization enabled optimizer](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.optimization.RegularizedOptimizer) implementation along with a `#!scala learn()` method which uses the optimizer member to calculate approximate values of the model parameters given the training data.

### Linear Models

Linear models; represented by the [`#!scala LinearModel[T, P, Q , R, S]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.LinearModel) trait are extensions of `#!scala ParameterizedLearner`, this top level trait is extended to yield many useful linear prediction models.

[_Generalized Linear Models_](core_glm.md) which are linear in parameters expression for the predictions $y$ given a vector of processed features $\phi(x)$ or basis functions.

$$
	\begin{equation}
		y = w^T\varphi(x) + \epsilon
	\end{equation}
$$


## Stochastic Processes

Stochastic processes (or random functions) are general probabilistic models which can be used to construct finite dimensional distributions over a set of sampled domain points. More specifically a stochastic process is a probabilistic function $f(.)$ defined on any _domain_ or _index set_ $\mathcal{X}$ such that for any finite collection $x_i \in \mathcal{X}, i = 1 \cdots N$, the finite dimensional distribution $P(f(x_1), \cdots, f(x_N))$ is coherently defined.

!!! tip
    The [`#!scala StochasticProcessModel[T, I, Y, W]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.StochasticProcessModel) trait extends `#!scala Model[T, I, Y]` and is the top level trait for the implementation of general stochastic processes. In order to extend it, one must implement among others a function to output the posterior predictive distribution `#!scala predictiveDistribution()`.

### Continuous Processes

By continuous processes, we mean processes whose values lie on a continuous domain (such as $\mathbb{R}^d$). The [`#!scala ContinuousProcessModel[T, I, Y, W]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.ContinuousProcessModel) abstract class provides a template which can be extended to implement continuous random process models.

!!! tip
    The [`#!scala ContinuousProcessModel`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.ContinuousProcessModel) class contains the method `predictionWithErrorBars()` which takes inputs test data and number of standard deviations, and generates predictions with upper and lower error bars around them. In order to create a sub-class of `#!scala ContinuousProcessModel`, you must implement the method `predictionWithErrorBars()`.

### Second Order Processes

Second order stochastic processes can be described by specifying the _mean_ (first order statistic) and _variance_ (second order statistic) of their finite dimensional distribution. The [`#!scala SecondOrderProcessModel[T, I, Y, K, M, W]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.SecondOrderProcessModel) trait is an abstract skeleton which describes what elements a second order process model must have i.e. the mean and covariance functions.


## Meta Models/Model Ensembles

Meta models use predictions from several candidate models and derive a prediction that is a meaningful combination of the individual predictions. This may be achieved in several ways some of which are.

* Average of predictions/voting
* Weighted predictions: Problem is now transferred to calculating appropriate weights.
* Learning some non-trivial functional transformation of the individual prediction, also known as _gating networks_.  

Currently the DynaML API has the following classes providing capabilities of meta models.

*Abstract Classes*

* [`#!scala MetaModel[D, D1, BaseModel]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.ensemble.MetaModel)
* [`#!scala CommitteeModel[D, D1, BaseModel]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.ensemble.CommitteeModel)

*Implementations*

* [LS-SVM Committee](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.svm.LSSVMCommittee)
* [Neural Committee](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.models.neuralnets.CommitteeNetwork)
