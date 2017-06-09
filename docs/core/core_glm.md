
_Generalized Linear Models_ (GLM) are available in the context of regression and binary classification, more specifically in DynaML the following members of the GLM family are implemented. The `#!scala GeneralizedLinearModel[T]` class is the base of the GLM hierarchy in DynaML, all linear models are extensions of it. It's companion object is used for the creation of GLM instances as follows.

```scala
val data: Stream[(DenseVector[Double], Double)] = ...

//The task variable is a string which is set to "regression" or "classification"
val task = ...

//The map variable defines a possibly higher dimensional function of the input
//which is akin to a basis function representation of the original features
val map: DenseVector[Double] => DenseVector[Double] = ...

//modeltype is set to "logit" or "probit"
//if one wishes to create a binary classification model,
//depending on the classification model involved
val modeltype = "logit"

val glm = GeneralizedLinearModel(data, task, map, modeltype)
```

## Normal GLM

The most common regression model, also known as _least squares linear regression_, implemented as the class `#!scala RegularizedGLM` which represents a regression model with the following prediction:

$$
	\begin{equation}
		y \ | \ \mathbf{x} \sim \mathcal{N}(w^T \varphi(\mathbf{x}), \sigma^{2})
	\end{equation}
$$

Here $\varphi(.)$ is an appropriately chosen set of _basis functions_. The inference problem is formulated as

$$
	\begin{equation}
		\min_{w} \ \mathcal{J}_P(w) = \frac{1}{2} \gamma \  w^Tw + \frac{1}{2} \sum_{k = 1}^{N} (y_k - w^T \varphi(x_k))^2
	\end{equation}
$$


## Logit GLM

In binary classification the most common GLM used is the _logistic regression_ model which is given by
$$
	\begin{equation}
		P(y \ = 1 \ | \ \mathbf{x}) = \sigma(w^T \varphi(\mathbf{x}) + b)
	\end{equation}
$$

Where $\sigma(z) = \frac{1}{1 + exp(-z)}$ is the logistic function which maps the output of the linear function $w^T \varphi(\mathbf{x}) + b$ to a probability value.

## Probit GLM

The _probit regression_ model is an alternative to the _logit_ model it is represented as:
$$
	\begin{equation}
		P(y \ = 1 \ | \ \mathbf{x}) = \Phi(w^T \varphi(\mathbf{x}) + b)
	\end{equation}
$$
Where $\Phi(z)$ is the cumulative distribution function of the standard normal distribution.
