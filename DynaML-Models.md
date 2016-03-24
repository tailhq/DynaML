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

#### Using Dual LSSVM in DynaML

```scala
val kernel = new ...
val data: Stream[(DenseVector[Double], Double)] = ... 
val model = new DLSSVM(data, data.length, kernel)
model.setRegParam(1.5).learn()
```

### Gaussian Processes

![gp]({{site.baseurl}}/public/gp.png)

<br/>

_Gaussian Processes_ are powerful non-parametric predictive models, which represent probability measures over spaces of functions. [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en) is the definitive guide on understanding their applications in machine learning and a gateway to their deeper theoretical foundations.

![gp-book]({{site.baseurl}}/public/gpbook.jpg)

<br/>

$$
	\begin{align}
		& y = f(x) + \epsilon \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
		& \left(\mathbf{y} \ \ \mathbf{f_*} \right)^T \sim \mathcal{N}\left(\mathbf{0}, \left[ \begin{matrix} K(X, X) + \sigma^{2} \it{I} & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{matrix} \right ] \right) 

	\end{align}
$$


In the presence of training data

$$
	X = (x_1, x_2, \cdot , x_n) \ y = (y_1, y_2, \cdot , y_n)
$$

Inference is carried out by calculating the posterior predictive distribution over the unknown targets

$$
	\mathbf{f_*}|X,\mathbf{y},X_*
$$

assuming $$ X_* $$, the test inputs are known. 

$$
	\begin{align}
		& \mathbf{f_*}|X,\mathbf{y},X_* \sim \mathcal{N}(\mathbf{\bar{f_*}}, cov(\mathbf{f_*}))  \label{eq:posterior}\\
		& \mathbf{\bar{f_*}} \overset{\triangle}{=} \mathbb{E}[\mathbf{f_*}|X,y,X_*] = K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1} \mathbf{y} \label{eq:posterior:mean} \\
		& cov(\mathbf{f_*}) = K(X_*,X_*) - K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1}K(X,X_*) 
	
	\end{align}
$$


#### Gaussian Processes in DynaML

```scala
val trainingdata: Stream[(DenseVector[Double], Double)] = ...
val kernel = new RBFKernel(2.5)
val noiseKernel = new DiracKernel(1.5)
val model = new GPRegression(kernel, noiseKernel, trainingData)
```

### Feed forward Neural Networks
