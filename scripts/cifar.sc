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
