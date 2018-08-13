!!! summary ""
    Version 1.5.3 of DynaML, released August 13, 2017, .


## Additions

### Tensorflow Integration
 
 **Package** `dynaml.tensorflow`
 
 **Training Stopping Criteria**
 
 Create common and simple training stop criteria such as.
 
  - Stop after fixed number of iterations `dtflearn.max_iter_stop(100000)`
  - Stop after change in value of loss goes below a threshold. `dtflearn.abs_loss_change_stop(0.0001)`
  - Stop after change in relative value of loss goes below a threshold. `dtflearn.rel_loss_change_stop(0.001)`


 **Neural Network Building Blocks** 
  
  - Added helper method ```dtlearn.build_tf_model()``` for training tensorflow models/estimators.

 **Usage**

 ```scala
 
 import io.github.mandar2812.dynaml.tensorflow._
 import org.platanios.tensorflow.api._
 import org.platanios.tensorflow.data.image.MNISTLoader
 import ammonite.ops._
 
 val tempdir = home/"tmp"
 
 val dataSet = MNISTLoader.load(java.nio.file.Paths.get(tempdir.toString()))
 val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
 val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
 val trainData =
   trainImages.zip(trainLabels)
     .repeat()
     .shuffle(10000)
     .batch(256)
     .prefetch(10)

 // Create the MLP model.
 val input = tf.learn.Input(UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2)))

 val trainInput = tf.learn.Input(UINT8, Shape(-1))

 val architecture = tf.learn.Flatten("Input/Flatten") >> 
   tf.learn.Cast("Input/Cast", FLOAT32) >>
   tf.learn.Linear("Layer_0/Linear", 128) >>  
   tf.learn.ReLU("Layer_0/ReLU", 0.1f) >>
   tf.learn.Linear("Layer_1/Linear", 64) >>
   tf.learn.ReLU("Layer_1/ReLU", 0.1f) >>
   tf.learn.Linear("Layer_2/Linear", 32) >>
   tf.learn.ReLU("Layer_2/ReLU", 0.1f) >>
   tf.learn.Linear("OutputLayer/Linear", 10)

  val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)

  val loss =
    tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.AdaGrad(0.1)

  // Directory in which to save summaries and checkpoints
  val summariesDir = java.nio.file.Paths.get((tempdir/"mnist_summaries").toString())


  val (model, estimator) = dtflearn.build_tf_model(
    architecture, input, trainInput, trainingInputLayer,
    loss, optimizer, summariesDir, dtflearn.max_iter_stop(1000),
    100, 100, 100)(trainData)

```

 - Build feedforward layers and feedforward layer stacks easier.

**Usage**

 ```scala

 import io.github.mandar2812.dynaml.tensorflow._
 import org.platanios.tensorflow.api._
 //Create a single feedforward layer

 val layer = dtflearn.feedforward(num_units = 10, useBias = true)(id = 1)

 //Create a stack of feedforward layers


 val net_layer_sizes = Seq(10, 5, 3)
 
 val stack = dtflearn.feedforward_stack(
   (i: Int) => dtflearn.Phi("Act_"+i), FLOAT64)(
   net_layer_sizes)

 ```
 
 
 #### Batch Normalisation
 
 [Batch normalisation](https://arxiv.org/abs/1502.03167) is used to standardize activations of convolutional layers and
 to speed up training of deep neural nets.
 
 **Usage**
 
 ```scala
 import io.github.mandar2812.dynaml.tensorflow._
  
 val bn = dtflearn.batch_norm("BatchNorm1")
 
 ```
 
 
 #### Inception v2
 
 The [_Inception_](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) architecture, proposed by Google is an important
 building block of _convolutional neural network_ architectures used in vision applications.
 
 ![inception](https://github.com/transcendent-ai-labs/DynaML/blob/master/docs/images/inception.png)
 
 DynaML now offers the Inception cell as a computational layer. 
 
 **Usage**
 
 ```scala
 import io.github.mandar2812.dynaml.pipes._
 import io.github.mandar2812.dynaml.tensorflow._
 import org.platanios.tensorflow.api._
 
 //Create an RELU activation, given a string name/identifier.
 val relu_act = DataPipe(tf.learn.ReLU(_))
 
 //Learn 10 filters in each branch of the inception cell
 val filters = Seq(10, 10, 10, 10)
 
 val inception_cell = dtflearn.inception_unit(
   channels = 3,  num_filters = filters, relu_act,
   //Apply batch normalisation after each convolution
   use_batch_norm = true)(layer_index = 1)
 
 ```
 
 In a subsequent [paper](https://arxiv.org/pdf/1512.00567.pdf), the authors introduced optimizations in the Inception 
 architecture, known colloquially as _Inception v2_.
 
 In _Inception v2_, larger convolutions (i.e. `3 x 3` and `5 x 5`) are implemented in a factorized manner 
 to reduce the number of parameters to be learned. For example the `3 x 3` convolution is expressed as a 
 combination of `1 x 3` and `3 x 1` convolutions.
 
 ![inception](https://github.com/transcendent-ai-labs/DynaML/blob/master/docs/images/conv-fact.png)
 
 Similarly the `5 x 5` convolutions can be expressed a combination of two `3 x 3` convolutions
 
 ![inception](https://github.com/transcendent-ai-labs/DynaML/blob/master/docs/images/conv-fact2.png)


### 3D Graphics 

**Package** [`dynaml.graphics`](https://github.com/transcendent-ai-labs/DynaML/blob/master/dynaml-core/src/main/scala-2.11/io/github/mandar2812/dynaml/graphics/plot3d/package.scala)

Create 3d plots of surfaces, for a use case, see the `jzydemo.sc` and `tf_wave_pde.sc`


### Library Organisation
 
 - Removed the `dynaml-notebook` module.
 
## Bugfixes

 
## Changes

