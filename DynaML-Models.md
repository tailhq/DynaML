---
layout: page
title: DynaML Models
---

## Model Classes

In DynaML all model implementations fit into a well defined class hierarchy. In fact every DynaML machine learning model is an extension of the abstract ```Model[T,Q,R]```. Lets go over the important classes of models, you can refer to the [wiki page](https://github.com/mandar2812/DynaML/wiki/Models) for more details.

### Parameterized Models

Many predictive models characterise the predictive process by a set of parameters which are used along with the data points to generate predictions, the ``` ParameterizedLearner``` class represents a skeleton for all parametric machine learning models such as _Generalized Linear Models_, _Neural Networks_, etc. 

### Linear Models

_Generalized Linear Models_ which are linear in parameters expression for the predictions $$y$$ given a vector of processed features $$\phi(x)$$ or basis functions.

$$
	\begin{equation}
		y = w^T\phi(x) + b + e
	\end{equation}
$$

### Non Parametric Models

Non parametric models generally grow with the size of the data set, some examples include _Gaussian Processes_ and _Dual LSSVM_ among others.


## Model Implementations

DynaML contains a few (but growing number) of algorithm implementations, currently the base data structures used for these implementations is `DenseVector` as defined in the [Breeze](https://github.com/scalanlp/breeze) linear algebra and statistics library.
The base model classes can be extended for specific applications like big data analysis where the programmer use customized data structures.

### Least Squares Support Vector Machines

Least Squares Support Vector Machines are a modification of the classical Support Vector Machine, please see the [book](http://www.amazon.com/Least-Squares-Support-Vector-Machines/dp/9812381511) for a complete background.

![lssvm-book]({{site.baseurl}}/public/cover_js_small.jpg)

In case of LSSVM regression one solves (by applying the [KKT](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) conditions) the following constrained optimization problem.

$$
	\begin{align}
		& \min_{w,b,e} \ \mathcal{J}_P(w,e) = \frac{1}{2}w^Tw + \gamma \frac{1}{2} \sum_{k = 1}^{N} e^2_k \\
		& y_k = w^T\phi(x) + b + e_k, \ k =1, \cdots, N 
	\end{align}
$$

Leading to a predictive model of the form.

$$
	\begin{equation}
		y(x) = \sum_{k = 1}^{N}\alpha_k K(x, x_k) + b
	\end{equation}
$$

Where the values $$\alpha \ \& \ b $$ are the solution of

$$
\begin{equation}
\left[\begin{array}{c|c}
   0  & 1^\intercal_v   \\ \hline
   1_v & K + \gamma^{-1} \mathit{I} 
\end{array}\right] 
\left[\begin{array}{c}
   b    \\ \hline
   \alpha  
\end{array}\right] = \left[\begin{array}{c}
   0    \\ \hline
   y  
\end{array}\right]
\end{equation}
$$

Here _K_ is the $$N \times N$$ kernel matrix whose entries are given by $$ K_{kl} = \phi(x_k)^\intercal\phi(x_l), \ \ k,l = 1, \cdots, N$$ and $$I$$ is the identity matrix of order $$N$$.


### Gaussian Processes

### Feed forward Neural Networks
