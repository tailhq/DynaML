!!! summary ""
    Version 1.5.3 of DynaML, released August 13, 2017, .


## Additions

### Tensorflow Integration
 
 
 **Package** `dynaml.tensorflow`
 
 #### Inception v2
 
 The [_Inception_](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) architecture, proposed by Google is an important
 building block of _convolutional neural network_ architectures used in vision applications.
 
 ![inception](https://github.com/transcendent-ai-labs/DynaML/blob/master/docs/images/inception.png)
 
 DynaML now offers the Inception cell as a computational layer. 
 
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
 

### Library Organisation
 
 - Removed the `dynaml-notebook` module.
 
## Bugfixes

 
## Changes

