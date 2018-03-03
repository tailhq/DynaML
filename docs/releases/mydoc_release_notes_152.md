!!! summary ""
    Version 1.5.2 of DynaML, released March 2, 2017, 
    introduces functionality through improvement in the pipes API and 
    increased integration with Tensorflow.


## Additions

 ### Tensorflow Integration
 
 - Tensorflow (beta) support now live, thanks to the [tensorflow_scala](https://github.com/eaplatanios/tensorflow_scala) project! Try it out in:
     * [CIFAR-10](https://github.com/transcendent-ai-labs/DynaML/blob/master/scripts/cifar.sc) example script
     * [MNIST](https://github.com/transcendent-ai-labs/DynaML/blob/master/scripts/mnist.sc) example script
 
 **Package** `dynaml.tensorflow`
 
 - The `dtf` package object houses utility functions related to tensorflow primitives. Currently supports creation of tensors from arrays.
 - The `dtflearn` package object deals with basic neural network building blocks which are often needed while constructing prediction architectures.
 

 ### Library Organisation
 
 - Added `dynaml-repl` and `dynaml-notebook` modules to repository.
 
 ### DynaML Server
 
 - DynaML ssh server now available 
   ```bash
   $ ./target/universal/stage/bin/dynaml --server
   ```
   To login to the server open a separate shell and type
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
