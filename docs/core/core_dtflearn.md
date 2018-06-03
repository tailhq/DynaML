!!! summary
    The [`dtflearn`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package) 
    object makes it easy to create and train neural networks of 
    varying complexity.



## Activation Functions

Apart from the activation functions defined in tensorflow for scala, DynaML defines some additional activations.

 - [Hyperbolic Tangent](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent) 
    ```scala 
    val act = dtflearn.Tanh("SomeIdentifier")
    ```
 - [Cumulative Gaussian](https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function)
    ```scala
    val act = dtflearn.Phi("OtherIdentifier")
    ```
 - [Generalized Logistic](https://en.wikipedia.org/wiki/Generalised_logistic_function)
    ```scala
    val act = dtflearn.GeneralizedLogistic("AnotherId")
    ```


## Layers

DynaML aims to supplement and extend the collection of layers available in `org.platanios.tensorflow.api.layers`, 
all the layers defined in DynaML's `tensorflow` package extend the `Layer[T, R]` class in 
`org.platanios.tensorflow.api.layers`.

### Radial Basis Function Network

_Radial Basis Function_ (RBF) networks are an important class of basis functions, each of which are expressed as 
decaying with distance from a defined central node.

$$
\begin{align}
f(x) & = \sum_{i}{w_{i} \varphi(||x - c_{i}||/\sigma)} \\
\varphi(u) & = exp(-u^2/2)
\end{align}
$$

The RBF layer implementation in DynaML treats the node center positions $c_i$ and length scales $\sigma_i$ as 
parameters to be learned via gradient based back-propagation.

```scala
import io.github.mandar2812.dynaml.tensorflow._

val rbf = dtflearn.rbf_layer(name = "rbf1", num_units = 10)
```

### Continuous Time RNN

Continuous time recurrent neural networks (CTRNN) are an important class of recurrent neural networks. They enable
the modelling of non-linear and potentially complex dynamical systems of multiple variables, with feedback.

Each state variable is modeled by a single neuron $y_i$, the evolution of the system $y = (y_1, \cdots, y_n)^T$ 
is governed by a set of coupled ordinary differential equations. These equations can be expressed in vector form as 
follows.

$$
\begin{align}
dy/dt & = - \Lambda . y + W . \sigma(G.y + b) \\ 
\end{align}
$$

The parameters of the system above are.

 - Time Constant/Decay Rate

$$
\begin{equation}
\Lambda = \begin{pmatrix}
        \lambda_1 & \cdots  & 0 \\ 
        \vdots & \ddots  & \vdots \\ 
        0 & \cdots  & \lambda_n  
        \end{pmatrix}
\end{equation}
$$

 - Gain

$$
\begin{equation}
G = \begin{pmatrix}
  g_{11} & \cdots  & g_{1n} \\ 
  \vdots & \ddots  & \vdots \\ 
  g_{n1} & \cdots  & g_{nn}  
  \end{pmatrix}
\end{equation}
$$

 - Bias

$$
\begin{equation}
b = \begin{pmatrix}
  b_{1}\\ 
  \vdots\\ 
  b_{n}  
  \end{pmatrix}
\end{equation}
$$

 - Weights

$$
\begin{equation}
W = \begin{pmatrix}
  w_{11} & \cdots  & w_{1n} \\ 
  \vdots & \ddots  & \vdots \\ 
  w_{n1} & \cdots  & w_{nn}  
  \end{pmatrix}
\end{equation}
$$

In order to use the CTRNN model in a modelling sequences of finite length, we need to solve its 
governing equations numerically. This gives us the trajectory of the state upto $T$ steps 
$y^{0}, \cdots, y^{T}$.

$$
y^{k+1} = y^{k} + \Delta t (- \Lambda . y^{k} + W . \sigma(G.y^{k} + b))
$$

DynaML's implementation of the CTRNN can be used to learn the trajectory of
dynamical systems upto a predefined time horizon. The parameters $\Lambda, G, b, W$ are
learned using gradient based loss minimization. 

The CTRNN implementations are also instances of `Layer[Output, Output]`, which take as input
tensors of shape $n$ and produce tensors of shape $(n, T)$, there are two variants that users
can choose from.

#### Fixed Time Step Integration

When the integration time step $\Delta t$ is user defined and fixed.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val ctrnn_layer = dtflearn.ctrnn(
name = "CTRNN_1", units = 10, 
horizon = 5, timestep = 0.1)
```

#### Dynamic Time Step Integration

When the integration time step $\Delta t$ is a parameter that can be learned during the
training process.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val dctrnn_layer = dtflearn.dctrnn(
name = "DCTRNN_1", units = 10, 
horizon = 5)
```


### Stack & Concatenate

Often one would need to combine inputs of previous layers in some manner, the following layers enable these operations.


#### Stack Inputs

This is a computational layer which performs the function of [`dtf.stack()`](/core/core_dtf/#stack).

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val stk_layer = dtflearn.stack_outputs("StackTensors", axis = 1)
```

#### Concatenate Inputs

This is a computational layer which performs the function of [`dtf.concatenate()`](/core/core_dtf/#concatenate).

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val concat_layer = dtflearn.stack_outputs("ConcatenateTensors", axis = 1)
```


#### Collect Layers

A sequence of layers can be collected into a single layer which accepts a sequence of symbolic tensors.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val layers = Seq(
  tf.learn.Linear("l1", 10),
  dtflearn.identity("Identity"),
  dtflearn.ctrnn(
    name = "CTRNN_1", units = 10, 
    horizon = 5, timestep = 0.1
  )
)

val combined_layer = dtflearn.stack_layers("Collect", layers)
```


#### Input Pairs 

To handle inputs consisting of pairs of elements, one can provide a separate layer for processing each of the elements.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val sl = dtflearn.tuple2_layer(
  "tuple2layer", 
  dtflearn.rbf_layer("rbf1", 10), 
  tf.learn.Linear("lin1", 10)) 
```

Combining the elements of Tuple2 can be done as follows.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

//Stack elements of the tuple into one tensor
val layer1 = dtflearn.stack_tuple2("tuple2layer", axis = 1) 
//Concatenate elements of the tuple into one tensor
val layer2 = dtflearn.concat_tuple2("tuple2layer", axis = 1) 
```

## Stoppage Criteria

In order to train tensorflow models using iterative gradient based models, the user must 
define some stoppage criteria for the training process. This can be done via the method 
`tf.learn.StopCriteria()`. The following preset stop criteria call `tf.learn.StopCriteria()` under the hood. 

### Iterations Based

```scala
val stopc1 = dtflearn.max_iter_stop(10000)
```

### Change in Loss

#### Absolute Value of Loss

```scala
val stopc2 = dtflearn.abs_loss_change_stop(0.1)
```

#### Relative Value of Loss

```scala
val stopc2 = dtflearn.rel_loss_change_stop(0.1)
```


## Network Building Blocks

To make it convenient to build deeper stacks of neural networks, DynaML includes some common layer design patterns
as ready made easy to use methods.

### Convolutional Neural Nets

[Convolutional neural networks](https://cs231n.github.io/convolutional-networks/) (CNN) are a crucial building block
of deep neural architectures for visual pattern recognition. It turns out that CNN layers must be combined with
other computational units such as _rectified linear_ (ReLU) activations, 
[dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) and _max pool_ layers.

Currently two abstractions are offered for building large CNN based network stacks

#### Convolutional Unit

A single CNN unit is expressed as a convolutional layer followed by a ReLU activation and proceeded by a dropout layer.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

//Learn 16 filters of shape (2, 2, 4), suitable for 4 channel jpeg images.
//Slide the filters over the image in steps of 1 pixel in each direction.
val cnn_unit = dtflearn.conv2d_unit(
    shape = Shape(2, 2, 4, 16), stride = (1, 1),
    relu_param = 0.05f, dropout = true,
    keep_prob = 0.55f)(i = 1)

```

#### Convolutional Pyramid

A CNN pyramid builds a stack of CNN units each with a stride multiplied by a factor of 2 and depth divided
by a factor of 2 with respect to the previous unit.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

//Start with a CNN unit of shape (2, 2, 3, 16) stride (1, 1)
//End with a CNN unit of shape (2, 2, 8, 4) and stride of (8, 8)
val cnn_stack = dtflearn.conv2d_pyramid(
  size = 2, num_channels_input = 3)(
  start_num_bits = 4, end_num_bits = 2)(
  relu_param = 0.1f, dropout = true, 
  keep_prob = 0.6F)

```


### Feed-forward Neural Nets

Feed-forward networks are the oldest and most frequently used components of neural network architectures, they are often
stacked into a number of layers. With `dtflearn.feedforward_stack()`, you can define feed-forward stacks of arbitrary
width and depth.

```scala

import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val net_layer_sizes = Seq(10, 20, 13, 15)

val architecture = dtflearn.feedforward_stack(
    (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
    net_layer_sizes)

```

## Building Tensorflow Models

After defining the key ingredients needed to build a tensorflow model, `dtflearn.build_tf_model()` builds a new 
computational graph and creates a tensorflow model and estimator which is trained on the provided data. In the 
following example, we bring together all the elements of model training: data, architecture, loss etc.


```scala
import ammonite.ops._
import io.github.mandar2812.dynaml.tensorflow.dtflearn
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.NN.SamePadding
import org.platanios.tensorflow.data.image.CIFARLoader
import java.nio.file.Paths


val tempdir = home/"tmp"

val dataSet = CIFARLoader.load(
  Paths.get(tempdir.toString()), 
  CIFARLoader.CIFAR_10)

val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)

val trainData = 
  trainImages.zip(trainLabels)
    .repeat()
    .shuffle(10000)
    .batch(128)
    .prefetch(10)


println("Building the classification model.")
val input = tf.learn.Input(
  UINT8, 
  Shape(
    -1, 
    dataSet.trainImages.shape(1), 
    dataSet.trainImages.shape(2), 
    dataSet.trainImages.shape(3))
)

val trainInput = tf.learn.Input(UINT8, Shape(-1))

val architecture = tf.learn.Cast("Input/Cast", FLOAT32) >>
  dtflearn.conv2d_pyramid(2, 3)(4, 2)(0.1f, true, 0.6F) >>
  tf.learn.MaxPool(
    "Layer_3/MaxPool", 
    Seq(1, 2, 2, 1), 
    1, 1, SamePadding) >>
  tf.learn.Flatten("Layer_3/Flatten") >>
  dtflearn.feedforward(256)(id = 4) >>
  tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
  dtflearn.feedforward(10)(id = 5)

val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)

val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
  tf.learn.Mean("Loss/Mean") >> 
  tf.learn.ScalarSummary("Loss/Summary", "Loss")

val optimizer = tf.train.AdaGrad(0.1)

println("Training the linear regression model.")
val summariesDir = java.nio.file.Paths.get(
  (tempdir/"cifar_summaries").toString()
)

val (model, estimator) = dtflearn.build_tf_model(
  architecture, input, trainInput, trainingInputLayer,
  loss, optimizer, summariesDir, 
  dtflearn.max_iter_stop(1000),
  100, 100, 100)(trainData)

```