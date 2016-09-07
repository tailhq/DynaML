---
title: Feedforward Neural Networks
tags: [neural_networks]
sidebar: coreapi_sidebar
permalink: core_ann.html
folder: coreapi
---

## Feed-forward Network

To create a feedforward network we need three entities.

* The training data (type parameter `D`)
* A data pipe which transforms the original data into a data structure that understood by the `FeedForwardNetwork`
* The network architecture (i.e. the network as a graph object)


### Network graph

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

### Model Building

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

The trained model can now be used for prediction, by using either the `predict()` method or the `feedForward()` value member both of which are members of `FeedForwardNetwork` (refer to the [api]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork) docs for more details).

```scala
val pattern = DenseVector(2.0, 3.5, 2.5)
val prediction = model.predict(pattern)
```

## Sparse Autoencoder

[Sparse autoencoders](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) are a feedforward architecture that are useful for unsupervised feature learning. They learn a compressed (or expanded) vector representation of the original data features. This process is known by various terms like _feature learning_, _feature engineering_, _representation learning_ etc. Autoencoders are amongst several models used for feature learning. Other notable examples include _convolutional neural networks_ (CNN), _principal component analysis_ (PCA), _Singular Value Decomposition_ (PCA) (a variant of  PCA), _Discrete Wavelet Transform_ (DWT), etc.

### Creation

Autoencoders can be created using the [```AutoEncoder```]({{site.baseurl}}/api_docs/dynaml-core/index.html#io.github.mandar2812.dynaml.models.neuralnets.AutoEncoder) class. Its constructor has the following arguments.


```scala
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._
import io.github.mandar2812.dynaml.optimization.BackPropagation

//Cast the training data as a stream of (x,x),
//where x are the DenseVector of features
val trainingData: Stream[(DenseVector[Double], DenseVector[Double])] = ...

val testData = ...

val enc = new AutoEncoder(
	inDim = trainingData.head._1.length,
	outDim = 4, acts = List(SIGMOID, LIN))
```

### Training

The training algorithm used is a modified version of standard back-propagation. The objective function can be seen as an addition of three terms.

$$
\begin{align}

\mathcal{J}(\mathbf{W}, \mathbf{X}; \lambda, \rho) &= \mathcal{L}(\mathbf{W}, \mathbf{X}) + \lambda \mathcal{R}(\mathbf{W}) + KL(\hat{\rho}\ ||\ \rho) \\
KL(\hat{\rho}\ ||\ \rho) &= \sum_{i = 1}^{n_h} \rho log(\frac{\rho}{\hat{\rho}_i}) + (1 - \rho) log(\frac{1-\rho}{1-\hat{\rho}_i}) \\
\hat{\rho}_i &= \frac{1}{m} \sum_{j = 1}^{N} a_{i}(x_j)

\end{align}
$$  

* $$\mathcal{L}(\mathbf{W}, \mathbf{X})$$ is the least squares loss.

* $$\mathcal{R}(\mathbf{W})$$ is the regularization penalty, with parameter $$\lambda$$.

* $$KL(\hat{\rho} \| \rho)$$ is the [_Kullback Leibler_](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) divergence, between the average activation (over all data instances $$x \in \mathbf{X}$$) of each hidden node and a specified value $$\rho \in [0,1]$$ which is also known as the _sparsity weight_.

```scala
//Set sparsity parameter for back propagation
BackPropagation.rho = 0.5

enc.optimizer
  .setRegParam(0.0)
  .setStepSize(1.5)
  .setNumIterations(200)
  .setMomentum(0.4)
  .setSparsityWeight(0.9)

enc.learn(trainingData.toStream)

val metrics = new MultiRegressionMetrics(
	testData.map(c => (enc.i(enc(c._1)), c._2)).toList,
	testData.length)

```

{% include links.html %}
