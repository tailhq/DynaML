{
  import ammonite.ops._

  import io.github.mandar2812.dynaml.tensorflow.dtflearn
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.ops.NN.SamePadding
  import org.platanios.tensorflow.data.image.CIFARLoader
  import java.nio.file.Paths


  val tempdir = home/"tmp"

  val dataSet = CIFARLoader.load(Paths.get(tempdir.toString()), CIFARLoader.CIFAR_10)
  val trainImages = tf.data.TensorSlicesDataset(dataSet.trainImages)
  val trainLabels = tf.data.TensorSlicesDataset(dataSet.trainLabels)
  val trainData =
    trainImages.zip(trainLabels)
      .repeat()
      .shuffle(10000)
      .batch(128)
      .prefetch(10)


  println("Building the logistic regression model.")
  val input = tf.learn.Input(
    UINT8, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2), dataSet.trainImages.shape(3))
  )

  val trainInput = tf.learn.Input(UINT8, Shape(-1))

  val architecture = tf.learn.Cast("Input/Cast", FLOAT32) >>
    dtflearn.conv2d_pyramid(2, 3)(4, 2)(0.1f, true, 0.6F) >>
    tf.learn.MaxPool("Layer_3/MaxPool", Seq(1, 2, 2, 1), 1, 1, SamePadding) >>
    tf.learn.Flatten("Layer_3/Flatten") >>
    dtflearn.feedforward(256)(id = 4) >>
    tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
    dtflearn.feedforward(10)(id = 5)

  val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)

  val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
    tf.learn.Mean("Loss/Mean") >> tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.AdaGrad(0.1)

  println("Training the linear regression model.")
  val summariesDir = java.nio.file.Paths.get((tempdir/"cifar_summaries").toString())

  val (model, estimator) = dtflearn.build_tf_model(
    architecture, input, trainInput, trainingInputLayer,
    loss, optimizer, summariesDir, dtflearn.max_iter_stop(1000),
    100, 100, 100)(trainData)

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
