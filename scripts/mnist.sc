{
  import io.github.mandar2812.dynaml.tensorflow.dtflearn
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.data.image.MNISTLoader
  import ammonite.ops._

  val tempdir = home/"tmp"

  // Load and batch data using pre-fetching.
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
    loss, optimizer, summariesDir, 1000)(trainData)

  def accuracy(images: Tensor, labels: Tensor): Float = {
    val predictions = estimator.infer(() => images)
    predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
  }


  val (trainAccuracy, testAccuracy) = (
    accuracy(dataSet.trainImages, dataSet.trainLabels),
    accuracy(dataSet.testImages, dataSet.testLabels))

  print("Train accuracy = ")
  pprint.pprintln(trainAccuracy)

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)
}
