!!! summary ""
    Version 1.5.2 of DynaML, released March 5, 2017, introduces functionality through improvement in the pipes API and increased integration with Tensorflow.


## Additions

### Tensorflow Integration
 
 - Tensorflow (beta) support now live, thanks to the [tensorflow_scala](https://github.com/eaplatanios/tensorflow_scala) project! Try it out in:
     * [CIFAR-10](https://github.com/transcendent-ai-labs/DynaML/blob/master/scripts/cifar.sc) example script
     * [MNIST](https://github.com/transcendent-ai-labs/DynaML/blob/master/scripts/mnist.sc) example script
 
 **Package** `dynaml.tensorflow`
 
 - The [`dtf`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5.2/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package$$dtf$) package object houses utility functions related to tensorflow primitives. Currently supports creation of tensors from arrays.
 
   ```scala 
   import io.github.tailhq.dynaml.tensorflow._
   import org.platanios.tensorflow.api._
   //Create a FLOAT32 Tensor of shape (2, 2), i.e. a square matrix
   val mat = dtf.tensor_f32(2, 2)(1d, 2d, 3d, 4d)
            
   //Create a random 2 * 3 matrix with independent standard normal entries
   val rand_mat = dtf.random(FLOAT32, 2, 3)(
     GaussianRV(0d, 1d) > DataPipe((x: Double) => x.toFloat)
   )
            
   //Multiply matrices
   val prod = mat.matmul(rand_mat)
   println(prod.summarize())
   
   val another_rand_mat = dtf.random(FLOAT32, 2, 3)(
     GaussianRV(0d, 1d) > DataPipe((x: Double) => x.toFloat)
   )
   
   //Stack tensors vertically, i.e. row wise
   val vert_tensor = dtf.stack(Seq(rand_mat, another_rand_mat), axis = 0)
   //Stack vectors horizontally, i.e. column wise
   val horz_tensor = dtf.stack(Seq(rand_mat, another_rand_mat), axis = 1)
   ```
 
 - The [`dtflearn`](https://transcendent-ai-labs.github.io/api_docs/DynaML/v1.5.2/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package$$dtflearn$) package object deals with basic neural network building blocks which are often needed while constructing prediction architectures.
 
   ```scala 
   //Create a simple neural architecture with one convolutional layer 
   //followed by a max pool and feedforward layer  
   val net = tf.learn.Cast("Input/Cast", FLOAT32) >>
     dtflearn.conv2d_pyramid(2, 3)(4, 2)(0.1f, true, 0.6F) >>
     tf.learn.MaxPool("Layer_3/MaxPool", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
     tf.learn.Flatten("Layer_3/Flatten") >>
     dtflearn.feedforward(256)(id = 4) >>
     tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
     dtflearn.feedforward(10)(id = 5)
   ```

### Library Organisation
 
 - Added `dynaml-repl` and `dynaml-notebook` modules to repository.
 
### DynaML Server
 
 - DynaML ssh server now available (only in Local mode)
   ```bash
   $ ./target/universal/stage/bin/dynaml --server
   ```
   To login to the server open a separate shell and type, (when prompted for password, just press ENTER)
   ```bash
   $ ssh repl@localhost -p22222
   ```

### Basis Generators
 - Legrendre polynomial basis generators
  
## Bugfixes

 - Acceptance rule of `HyperParameterMCMC` and related classes.

## Changes

 - Increased pretty printing to screen instead of logging.


## Cleanup

**Package** `dynaml.models.svm`
 - Removal of deprecated model classes from `svm` package
