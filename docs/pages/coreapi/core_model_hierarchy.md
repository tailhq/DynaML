---
title: Introduction to the Model Hierarchy
sidebar: coreapi_sidebar
permalink: core_model_hierarchy.html
folder: coreapi
---


## Model Classes

In DynaML all model implementations fit into a well defined class hierarchy. In fact every DynaML machine learning model is an extension of the ```Model[T,Q,R]``` trait. The ```Model``` trait is quite bare bones: machine learning models are viewed as objects containing two parts or components.

  1. A training data set (of type ```T```).  

  2. A method ```predict(point: Q):R``` to generate a prediction of type ```R``` given a data point of type ```Q```.


## Parameterized Models

Many predictive models calculate predictions by formulating an expression which includes a set of parameters which are used along with the data points to generate predictions, the ``` ParameterizedLearner[G, T, Q, R, S]``` class represents a skeleton for all parametric machine learning models such as [_Generalized Linear Models_]({{site.baseurl}}/core_glm.html), [_Neural Networks_]({{site.baseurl}}/core_ann.html), etc.

The defining characteristic of classes which extend ```ParameterizedLearner``` is that they must contain a member variable ```optimizer: RegularizedOptimizer[T, Q, R, S]``` which represents a regularization enabled optimizer implementation along with a ```learn()``` method which uses the optimizer member to calculate approximate values of the model parameters given the training data.

### Linear Models

Linear models; represented by the ```LinearModel[T, P, Q , R, S]``` trait are extensions of ```ParameterizedLearner```, this top level trait is extended to yield many useful linear prediction models.

[_Generalized Linear Models_]({{site.baseurl}}/core_glm.html) which are linear in parameters expression for the predictions $$y$$ given a vector of processed features $$\phi(x)$$ or basis functions.

$$
	\begin{equation}
		y = w^T\varphi(x) + \epsilon
	\end{equation}
$$

### Neural Networks

<br/>

![feedforward-NN]({{site.baseurl}}/images/fnn.png)

<br/>

Feed forward neural networks are the most common network architectures in predictive modeling, DynaML has an implementation of feed forward architectures that is trained using _Backpropogation_ with momentum.

In a feed forward neural network with a single hidden layer the predicted target $$y$$ is expressed using the edge weights and node values in the following manner (this expression is easily extended for multi-layer nets).

$$
	\begin{equation}
		y = W_2 \sigma(W_1 \mathbf{x} + b_1) + b_2
	\end{equation}
$$

Where $$W_1 , \ W_2$$  are matrices representing edge weights for the hidden layer and output layer respectively and $$\sigma(.)$$ represents a monotonic _activation_ function, the usual choices are _sigmoid_, _tanh_, _linear_ or _rectified linear_ functions.

## Non Parametric Models

Non parametric models generally grow with the size of the data set, some examples include _Gaussian Processes_ and _Dual LSSVM_ among others.

### LSSVM

Least Squares Support Vector Machines are a modification of the classical Support Vector Machine, please see [Suykens et. al](http://www.amazon.com/Least-Squares-Support-Vector-Machines/dp/9812381511) for a complete background.

![lssvm-book]({{site.baseurl}}/images/cover_js_small.jpg)


### Stochastic Processes

Stochastic processes are general probabilistic models which can be used to construct finite dimensional distributions over a set of sampled domain points. More specifically a stochastic process is a probabilistic function $$f(.)$$ defined on any _domain_ or _index set_ $$\mathcal{X}$$ such that for any finite collection $$x_i \in \mathcal{X}, i = 1 \cdots N$$, the finite dimensional distribution $$P(f(x_1), \cdots, f(x_N))$$ is coherently defined.

In DynaML the ```StochasticProcess[T, I, Y, W]``` trait extends ```Model[T, I, Y]``` and is the top level trait for the implementation of general stochastic processes. In order to extend it, one must implement among others a function to output the posterior predictive distribution ```predictiveDistribution()```.

### Gaussian Processes

![gp]({{site.baseurl}}/images/gp.png)

<br/>
_Gaussian Processes_ are stochastic processes whose finite dimensional distributions are multivariate gaussians.

_Gaussian Processes_ are powerful non-parametric predictive models, which represent probability measures over spaces of functions. [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en) is the definitive guide on understanding their applications in machine learning and a gateway to their deeper theoretical foundations.

![gp-book]({{site.baseurl}}/images/gpbook.jpg)

<br/>

### Student T Processes

Student T processes are generalizations of Gaussian Processes, in this case the finite dimensional distribution on a set of points is a multivariate t distribution.
