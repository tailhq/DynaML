{
  import ammonite.ops._

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
      .batch(64)
      .prefetch(10)


  println("Building the logistic regression model.")
  val input = tf.learn.Input(
    UINT8,
    Shape(
      -1,
      dataSet.trainImages.shape(1),
      dataSet.trainImages.shape(2),
      dataSet.trainImages.shape(3))
  )

  val trainInput = tf.learn.Input(UINT8, Shape(-1))

  val layer = tf.learn.Cast(FLOAT32) >>
    tf.learn.Conv2D(Shape(2, 2, 3, 16), 1, 1, SamePadding, name = "Conv2D_0") >>
    tf.learn.AddBias(name = "Bias_0") >>
    tf.learn.ReLU(0.1f) >>
    tf.learn.MaxPool(Seq(1, 2, 2, 1), 1, 1, SamePadding, name = "MaxPool_0") >>
    tf.learn.Conv2D(Shape(2, 2, 16, 32), 1, 1, SamePadding, name = "Conv2D_1") >>
    tf.learn.AddBias(name = "Bias_1") >>
    tf.learn.ReLU(0.1f) >>
    tf.learn.MaxPool(Seq(1, 2, 2, 1), 1, 1, SamePadding, name = "MaxPool_1") >>
    tf.learn.Flatten() >>
    tf.learn.Linear(256, name = "Layer_2") >> tf.learn.ReLU(0.1f) >>
    tf.learn.Linear(10, name = "OutputLayer")

  val trainingInputLayer = tf.learn.Cast(INT64)
  val loss = tf.learn.SparseSoftmaxCrossEntropy() >> tf.learn.Mean() >> tf.learn.ScalarSummary("Loss")
  val optimizer = tf.train.AdaGrad(0.1)

  val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

  println("Training the linear regression model.")
  val summariesDir = java.nio.file.Paths.get((tempdir/"cifar_summaries").toString())
  val estimator = tf.learn.InMemoryEstimator(
    model,
    tf.learn.Configuration(Some(summariesDir)),
    tf.learn.StopCriteria(maxSteps = Some(100000)),
    Set(
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
      tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
    tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 100))
  estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(500)))

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
