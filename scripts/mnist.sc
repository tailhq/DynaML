{
  import _root_.java.nio.file.Paths
  import _root_.io.github.tailhq.dynaml.pipes._
  import _root_.io.github.tailhq.dynaml.tensorflow._
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.data.image.MNISTLoader
  import _root_.ammonite.ops._

  // Load and batch data using pre-fetching.
  val tempdir = home/"tmp"

  val dataSet = MNISTLoader.load(Paths.get(tempdir.toString()), MNISTLoader.MNIST)

  val dtf_cifar_data = dtfdata.tf_dataset(
    dtfdata.supervised_dataset(
      dataSet.trainImages.unstack(axis = 0),
      dataSet.trainLabels.castTo[Long].unstack(axis = -1)),
    dtfdata.supervised_dataset(
      dataSet.testImages.unstack(axis = 0),
      dataSet.testLabels.castTo[Long].unstack(axis = -1))
  )

  val architecture = tf.learn.Cast[UByte, Float]("Input/Cast") >>
    tf.learn.Flatten[Float]("Input/Flatten") >>
    tf.learn.Linear[Float]("Layer_0/Linear", 128) >>
    tf.learn.ReLU[Float]("Layer_0/ReLU", 0.1f) >>
    tf.learn.Linear[Float]("Layer_1/Linear", 64) >>
    tf.learn.ReLU[Float]("Layer_1/ReLU", 0.1f) >>
    tf.learn.Linear[Float]("Layer_2/Linear", 32) >>
    tf.learn.ReLU[Float]("Layer_2/ReLU", 0.1f) >>
    tf.learn.Linear[Float]("OutputLayer/Linear", 10)

  val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss/CrossEntropy") >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.Adam(0.1f)

  val mnist_model = dtflearn.model[
    Output[UByte], Output[Long], Output[Float], Float,
    Tensor[UByte], UINT8, Shape,
    Tensor[Long], INT64, Shape,
    Tensor[Float], FLOAT32, Shape](
    architecture,
    (UINT8, dataSet.trainImages.shape(1::)),
    (INT64, Shape()),
    loss
  )

  val data_ops = dtflearn.model.data_ops(
    shuffleBuffer = 5000,
    batchSize = 128,
    prefetchSize = 10,
    concatOpI = Some(dtfpipe.EagerStack[UByte]()),
    concatOpT = Some(dtfpipe.EagerStack[Long]()),
    concatOpO = Some(dtfpipe.EagerConcatenate[Float]())
  )

  val config = dtflearn.model.trainConfig(
    tempdir/"mnist_summaries",
    data_ops,
    optimizer,
    dtflearn.rel_loss_change_stop(0.05, 500),
    Some(
      dtflearn.model._train_hooks(
        tempdir/"mnist_summaries",
        stepRateFreq = 100,
        summarySaveFreq = 100,
        checkPointFreq = 100)
    ))



  mnist_model.train(dtf_cifar_data.training_dataset, config)

  def accuracy(predictions: Tensor[Long], labels: Tensor[Long]): Float =
    tfi.equal(predictions.argmax[Long](1), labels)
      .castTo[Float]
      .mean()
      .scalar
      .asInstanceOf[Float]

  val (trainingPreds, testPreds): (Tensor[Float], Tensor[Float]) = (
    mnist_model.infer_batch(dtf_cifar_data.training_dataset.map(p => p._1), data_ops).left.get,
    mnist_model.infer_batch(dtf_cifar_data.test_dataset.map(p => p._1), data_ops).left.get
  )

  val (trainAccuracy, testAccuracy) = (
    accuracy(trainingPreds.castTo[Long], dataSet.trainLabels.castTo[Long]),
    accuracy(testPreds.castTo[Long], dataSet.testLabels.castTo[Long]))

  print("Train accuracy = ")
  pprint.pprintln(trainAccuracy)

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)

}
