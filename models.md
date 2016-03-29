---
layout: page
title: Models
noToc: true
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


-----

## Model Implementations

DynaML contains a few (but growing number) of algorithm implementations, currently the base data structures used for these implementations is `DenseVector` as defined in the [Breeze](https://github.com/scalanlp/breeze) linear algebra and statistics library.
The base model classes can be extended for specific applications like big data analysis where the programmer use customized data structures.



### Regularized Least Squares

The [_regularized least squares_](https://en.wikipedia.org/wiki/Tikhonov_regularization) model builds a predictor of the following form (assuming the data is centered see the data pipes [page]({{site.baseurl}}/data-pipes/#traintestgaussianstandardization)).

$$
	y = w^T \varphi(x)
$$

Here $$\varphi(.)$$ is an appropriately chosen set of _basis functions_. The inference problem is formulated as.

$$
	\begin{equation}
		\min_{w} \ \mathcal{J}_P(w) = \frac{1}{2} \gamma \  w^Tw + \frac{1}{2} \sum_{k = 1}^{N} (y_k - w^T \varphi(x_k))^2
	\end{equation}
$$


#### Regularized Least Squares in DynaML

```scala
val basisFunc: (DenseVector[Double]) => DenseVector[Double] = ...
val data: Stream[(DenseVector[Double], Double)] = ... 
val model = new RegularizedGLM(data, data.length, basisFunc)
model.setRegParam(1.5).learn()
```



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

-----

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

To learn more about extending the Gaussian Process base classes/traits refer to the [wiki](https://github.com/mandar2812/DynaML/wiki/Gaussian-Processes).

-----

### Feed forward Neural Networks

<br/>

![feedforward-NN]({{site.baseurl}}/public/fnn.png)

<br/>

Feed forward neural networks are the most common network architectures in predictive modeling, DynaML has an implementation of feed forward architectures that is trained using _Backpropogation_ with momentum.

In a feed forward neural network with a single hidden layer the predicted target $$y$$ is expressed using the edge weights and node values in the following manner (this expression is easily extended for multi-layer nets).

$$
	\begin{equation}
		y = W_2 \sigma(W_1 \mathbf{x} + b_1) + b_2
	\end{equation}
$$

Where $$W_1 , \ W_2$$  are matrices representing edge weights for the hidden layer and output layer respectively and $$\sigma(.)$$ represents a monotonic _activation_ function, the usual choices are _sigmoid_, _tanh_, _linear_ or _rectified linear_ functions.

#### Feed forward nets in DynaML

To create a feedforward network we need three entities.

* The training data (type parameter `D`)
* A data pipe which transforms the original data into a data structure that understood by the `FeedForwardNetwork`
* The network architecture (i.e. the network as a graph object)


A standard feedforward network can be created by first initializing the network architecture/graph.

```scala
val gr = FFNeuralGraph(num_inputs = 3, num_outputs = 1, 
hidden_layers = 1, List("logsig", "linear"), List(5))
```

This creates a neural network graph with one hidden layer, 3 input nodes, 1 output node and assigns sigmoid activation in the hidden layer. It also creates 5 neurons in the hidden layer.

Next we create a data transform pipe which converts instances of the data input-output patterns to `(DenseVector[Double], DenseVector[Double])`, this is required in many data processing applications where the data structure storing the training data is not a [breeze](https://github.com/scalanlp/breeze) vector.

Lets say we have data in the form `trainingdata: Stream[(DenseVector[Double], Double)]`, i.e. we have input features as breeze vectors and scalar output values which help the network learn an unknown function. We can write the transform as.

```scala
val transform = DataPipe(
	(d: Stream[(DenseVector[Double], Double)]) =>
		d.map(el => (el._1, DenseVector(el._2)))
)
```

We are now in a position to initialize a feed forward neural network model.

```scala
val model = new FeedForwardNetwork[
	Stream[(DenseVector[Double], Double)]
](trainingdata, gr, transform)
```

Here the variable `trainingdata` represents the training input output pairs, which must conform to the type argument given in square brackets (i.e. `Stream[(DenseVector[Double], Double)]`).

Training the model using back propagation can be done as follows, you can set custom values for the backpropagation parameters like the learning rate, momentum factor, mini batch fraction, regularization and number of learning iterations.

```scala
model.setLearningRate(0.09)
   .setMaxIterations(100)
   .setBatchFraction(0.85)
   .setMomentum(0.45)
   .setRegParam(0.0001)
   .learn()
```

The trained model can now be used for prediction, by using either the `predict()` method or the `feedForward()` value member both of which are members of `FeedForwardNetwork` (refer to the [api](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork) docs for more details).

```scala
val pattern = DenseVector(2.0, 3.5, 2.5)
val prediction = model.predict(pattern)
```

-----

### Neural Committee Models

Quite often it is observed that one cannot represent an unknown function accurately with a single neural network, it is then beneficial to consider a _committee_ of neural nets, each of which is trained on the entire or subsampled data. The final prediction of the model is a weighted average of the predictions of all the models in the committee.

#### Neural Committees in DynaML

```scala
val trainTestData: Stream[(DenseVector[Double], Double)] = .... 

val configs =
  for (c <- List(3, 5, 7, 9); a <- List("tansig", "logsig"))
  yield(c,a)

val networks = configs.map(couple => {
  FFNeuralGraph(trainTest._1._1.head._1.length, 1, 1,
    List(couple._2, "linear"),List(couple._1))
   })

val transform =
	DataPipe((d: Stream[(DenseVector[Double], Double)]) =>
	d.map(el => (el._1, DenseVector(el._2))))

val model =
  new CommitteeNetwork[Stream[(DenseVector[Double], Double)]](
    trainTestData, transform, networks:_*
  )

model.setLearningRate(0.05)
  .setMaxIterations(100)
  .setBatchFraction(0.8)
  .setMomentum(0.35)
  .setRegParam(0.0005)
  .learn()
		  
```

