!!! summary
    The [`dtflearn`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package$$dtflearn$) 
    object makes it easy to create and train neural networks of 
    varying complexity.


The `dtflearn` object contains various utilities for the creation and training of neural networks. These are.

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

### Continuous Time RNN

#### Fixed Time Step Integration

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val ctrnn_layer = dtflearn.ctrnn(
name = "CTRNN_1", units = 10, 
horizon = 5, timestep = 0.1)
```

#### Dynamic Time Step Integration

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._ 

val dctrnn_layer = dtflearn.dctrnn(
name = "DCTRNN_1", units = 10, 
horizon = 5)
```

### Radial Basis Function Network

```scala
import io.github.mandar2812.dynaml.tensorflow._

val rbf = dtflearn.rbf_layer("rbf1", 10)
```

### Stack & Concatenate

### Stack Outputs

## Layers for Working with Tuple2



#### Tuple 

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val sl = dtflearn.tuple2_layer(
  "tuple2layer", 
  dtflearn.rbf_layer("rbf1", 10), 
  tf.learn.Linear("lin1", 10)) 
```

#### Combine Elements of Tuple2


## Stoppage Criteria



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

### Convolutional Neural Nets

### Feed-forward Neural Nets

## Building Tensorflow Models