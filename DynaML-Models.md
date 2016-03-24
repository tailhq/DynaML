---
layout: page
title: DynaML Models
---

## Model Classes

In DynaML all model implementations fit into a well defined class hierarchy. In fact every DynaML machine learning model is an extension of the abstract ```Model[T,Q,R]``` see the github [wiki page](https://github.com/mandar2812/DynaML/wiki/Models) for more details. Lets go over the important classes of models.

### Parameterized Models

Many predictive models characterise the predictive process by a set of parameters which are used along with the data points to generate predictions, the ``` ParameterizedLearner``` class represents a skeleton for all parametric machine learning models such as _Generalized Linear Models_, _Neural Networks_, etc. Refer to the [wiki page](https://github.com/mandar2812/DynaML/wiki/Models) for more details.

### Linear Models

_Generalized Linear Models_ which are linear in parameters expression for the predictions $$y$$ given a vector of processed features $$\phi(x)$$ or basis functions.

$$
	\begin{equation}
		y = w^T\phi(x) + b + e
	\end{equation}
$$

### Non Parametric Models




## Model Implementations

### Regression

#### Least Squares Support Vector Machines

#### Gaussian Processes

#### Feed forward Neural Networks
