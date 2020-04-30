![3dplot](docs-old/images/dynaml_logo3.png)

# DynaML: ML + JVM + Scala


[![Join the chat at https://gitter.im/DynaML/Lobby](https://badges.gitter.im/DynaML/Lobby.svg)](https://gitter.im/DynaML/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/transcendent-ai-labs/DynaML.svg?branch=master)](https://travis-ci.org/transcendent-ai-labs/DynaML) [![](https://jitpack.io/v/transcendent-ai-labs/DynaML.svg)](https://jitpack.io/#transcendent-ai-labs/DynaML)
[![Coverage Status](https://coveralls.io/repos/github/transcendent-ai-labs/DynaML/badge.svg)](https://coveralls.io/github/transcendent-ai-labs/DynaML)

------------------

<br/>

DynaML is a Scala & JVM Machine Learning toolbox for research, education & industry.

<br/>

<table>
    <tr>
        <th> <img src="docs-old/images/plot3dsmall.jpeg" alt="Plot3d"> </th> 
        <th> <img src="docs-old/images/plots_small.png" alt="Plot2d"> </th>
    </tr>
</table>


------------------

## Motivation


 - __Interactive.__ Don't want to create Maven/sbt project skeletons
 every time you want to try out ideas? Create and execute [scala worksheets](scripts/randomvariables.sc) 
 in the DynaML shell. DynaML comes packaged with a customized version of the [Ammonite](http://ammonite.io) REPL, 
 with *auto-complete*, file operations and scripting capabilities.  
 
 - __End to End.__ Create complex pre-processing pipelines with the [data pipes](https://transcendent-ai-labs.github.io/DynaML/pipes/pipes/) API, 
 train models ([deep nets](scripts/cifar.sc), [gaussian processes](https://transcendent-ai-labs.github.io/DynaML/core/core_gp/), 
 [linear models](https://transcendent-ai-labs.github.io/DynaML/core/core_glm/) and more), 
 optimize over [hyper-parameters](https://transcendent-ai-labs.github.io/DynaML/core/core_opt_global/), 
 [evaluate](https://transcendent-ai-labs.github.io/DynaML/core/core_model_evaluation/) model predictions and 
 [visualise](https://transcendent-ai-labs.github.io/DynaML/core/core_graphics/) results.
 
 - __Enterprise Friendly.__ Take advantage of the JVM and Scala ecosystem, use Apache [Spark](https://spark.apache.org) 
 to write scalable data analysis jobs, [Tensorflow](http://tensorflow.org) for deep learning, all in the same toolbox.

------------------

## Getting Started

### Platform Compatibility

Currently, only *nix and OSX platforms are supported.

DynaML is compatible with Scala `2.12`

### Installation

Easiest way to install DynaML is cloning & compiling from the [github](/) repository. Please take a look at 
the [installation](https://transcendent-ai-labs.github.io/DynaML/installation/installation/) instructions in the 
[user guide](https://transcendent-ai-labs.github.io/DynaML/), to make sure that you have the pre-requisites 
and to configure your installation.

------------------

## CIFAR in under 200 lines

Below is a sample [script](scripts/cifar.sc) where we train a neural network of stacked 
[Inception](https://arxiv.org/pdf/1409.4842.pdf) cells on the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)
image classification task.

```scala
{
  import _root_.ammonite.ops._
  import _root_.io.github.mandar2812.dynaml.pipes.DataPipe
  import _root_.io.github.mandar2812.dynaml.tensorflow.{
    dtflearn,
    dtfutils,
    dtfdata,
    dtfpipe
  }
  import _root_.org.platanios.tensorflow.api._
  import _root_.org.platanios.tensorflow.api.learn.layers.Activation
  import _root_.org.platanios.tensorflow.data.image.CIFARLoader
  import _root_.java.nio.file.Paths

  val tempdir = home / "tmp"

  val dataSet =
    CIFARLoader.load(Paths.get(tempdir.toString()), CIFARLoader.CIFAR_10)

  val dtf_cifar_data = dtfdata.tf_dataset(
    dtfdata.supervised_dataset(
      dataSet.trainImages.unstack(axis = 0),
      dataSet.trainLabels.castTo[Long].unstack(axis = -1)
    ),
    dtfdata.supervised_dataset(
      dataSet.testImages.unstack(axis = 0),
      dataSet.testLabels.castTo[Long].unstack(axis = -1)
    )
  )

  println("Building the model.")

  val relu_act =
    DataPipe[String, Activation[Float]]((x: String) => tf.learn.ReLU[Float](x))

  val architecture =
    tf.learn.Cast[UByte, Float]("Input/Cast") >>
      dtflearn.inception_unit[Float](channels = 3, Seq.fill(4)(10), relu_act)(
        layer_index = 1
      ) >>
      dtflearn.inception_unit[Float](channels = 40, Seq.fill(4)(5), relu_act)(
        layer_index = 2
      ) >>
      tf.learn.Flatten[Float]("Layer_3/Flatten") >>
      dtflearn.feedforward[Float](256)(id = 4) >>
      tf.learn.ReLU[Float]("Layer_4/ReLU", 0.1f) >>
      dtflearn.feedforward[Float](10)(id = 5)

  val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float](
    "Loss/CrossEntropy"
  ) >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.Adam(0.1f)

  val cifar_model =
    dtflearn.model[
      Output[UByte], Output[Long], Output[Float], Float, 
      Tensor[UByte], UINT8, Shape, 
      Tensor[Long], INT64, Shape, 
      Tensor[Float], FLOAT32, Shape](
      architecture,
      (UINT8, dataSet.trainImages.shape(1 ::)),
      (INT64, Shape()),
      loss
    )

  val data_ops = dtflearn.model.data_ops[(Output[UByte], Output[Long])](
    shuffleBuffer = 5000,
    batchSize = 128,
    prefetchSize = 10
  )

  val train_config = dtflearn.model.trainConfig(
    tempdir / "cifar_summaries",
    data_ops,
    optimizer,
    dtflearn.rel_loss_change_stop(0.05, 500),
    Some(
      dtflearn.model._train_hooks(
        tempdir / "cifar_summaries",
        stepRateFreq = 100,
        summarySaveFreq = 100,
        checkPointFreq = 100
      )
    )
  )

  val pattern_to_tensor =
    DataPipe[Seq[(Tensor[UByte], Tensor[Long])], (Tensor[UByte], Tensor[Long])](
      ds => {
        val (xs, ys) = ds.unzip

        (
          dtfpipe.EagerStack[UByte](axis = 0).run(xs),
          dtfpipe.EagerStack[Long](axis = 0).run(ys)
        )
      }
    )

  val data_handle_ops = dtflearn.model.tf_data_handle_ops[
    (Tensor[UByte], Tensor[Long]),
    (Tensor[UByte], Tensor[Long]),
    Tensor[Float],
    (Output[UByte], Output[Long])
  ](
    bufferSize = 500,
    patternToTensor = Some(pattern_to_tensor),
    concatOpO = Some(dtfpipe.EagerConcatenate[Float]())
  )

  val data_handle_ops_infer =
    dtflearn.model.tf_data_handle_ops[Tensor[UByte], Tensor[UByte], Tensor[
      Float
    ], Output[UByte]](
      bufferSize = 1000,
      patternToTensor = Some(dtfpipe.EagerStack[UByte](axis = 0)),
      concatOpO = Some(dtfpipe.EagerConcatenate[Float]())
    )

  cifar_model.train(
    dtf_cifar_data.training_dataset,
    train_config,
    data_handle_ops
  )

  def accuracy(predictions: Tensor[Long], labels: Tensor[Long]): Float =
    tfi
      .equal(predictions.argmax[Long](1), labels)
      .castTo[Float]
      .mean()
      .scalar
      .asInstanceOf[Float]

  val (trainingPreds, testPreds): (Tensor[Float], Tensor[Float]) = (
    cifar_model
      .infer_batch(
        dtf_cifar_data.training_dataset.map(p => p._1),
        data_handle_ops_infer
      )
      .left
      .get,
    cifar_model
      .infer_batch(
        dtf_cifar_data.test_dataset.map(p => p._1),
        data_handle_ops_infer
      )
      .left
      .get
  )

  val (trainAccuracy, testAccuracy) = (
    accuracy(trainingPreds.castTo[Long], dataSet.trainLabels.castTo[Long]),
    accuracy(testPreds.castTo[Long], dataSet.testLabels.castTo[Long])
  )

  print("Train accuracy = ")
  pprint.pprintln(trainAccuracy)

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)
}
```

------------------


## Support & Community

 - [User guide](https://transcendent-ai-labs.github.io/DynaML/)
 - [Gitter](https://gitter.im/DynaML/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
 - [Contributing](https://github.com/transcendent-ai-labs/DynaML/blob/master/CONTRIBUTING.md)
 - [Code of Conduct](https://github.com/transcendent-ai-labs/DynaML/blob/master/CODE_OF_CONDUCT.md)
