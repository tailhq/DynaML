{
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
  val layer = tf.learn.Flatten() >>
    tf.learn.Cast(FLOAT32) >>
    tf.learn.Linear(128, name = "Layer_0") >> tf.learn.ReLU(0.1f) >>
    tf.learn.Linear(64, name = "Layer_1") >> tf.learn.ReLU(0.1f) >>
    tf.learn.Linear(32, name = "Layer_2") >> tf.learn.ReLU(0.1f) >>
    tf.learn.Linear(10, name = "OutputLayer")
  val trainingInputLayer = tf.learn.Cast(INT64)
  val loss = tf.learn.SparseSoftmaxCrossEntropy() >> tf.learn.Mean() >> tf.learn.ScalarSummary("Loss")
  val optimizer = tf.train.AdaGrad(0.1)
  val model = tf.learn.Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

  val summariesDir = java.nio.file.Paths.get((tempdir/"summaries").toString())                 // Directory in which to save summaries and checkpoints

  val estimator = tf.learn.InMemoryEstimator(
    model,
    tf.learn.Configuration(Some(summariesDir)),
    tf.learn.StopCriteria(maxSteps = Some(100000)),
    Set(
      tf.learn.StepRateHook(log = false, summaryDirectory = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
      tf.learn.SummarySaverHook(summariesDir, tf.learn.StepHookTrigger(100)),
      tf.learn.CheckpointSaverHook(summariesDir, tf.learn.StepHookTrigger(100))),
    tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1))
  estimator.train(() => trainData, tf.learn.StopCriteria(maxSteps = Some(1000)))

  def accuracy(images: Tensor, labels: Tensor): Float = {
    val predictions = estimator.infer(() => images)
    predictions.argmax(1).cast(UINT8).equal(labels).cast(FLOAT32).mean().scalar.asInstanceOf[Float]
  }

  println(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
  println(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
}
