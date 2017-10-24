{
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.tf.learn._
  import org.platanios.tensorflow.data.loaders.MNISTLoader
  import ammonite.ops._

  val tempdir = home/"tmp"

  // Load and batch data using pre-fetching.
  val dataSet = MNISTLoader.load(java.nio.file.Paths.get(tempdir.toString()))
  val trainImages = DatasetFromSlices(dataSet.trainImages)
  val trainLabels = DatasetFromSlices(dataSet.trainLabels)
  val trainData =
    trainImages.zip(trainLabels)
      .repeat()
      .shuffle(10000)
      .batch(256)
      .prefetch(10)

  // Create the MLP model.
  val input = Input(UINT8, Shape(-1, 28, 28))
  val trainInput = Input(UINT8, Shape(-1))
  val layer = Flatten() >> Cast(FLOAT32) >>
    Linear(128, name = "Layer_0") >> ReLU(0.1f) >>
    Linear(64, name = "Layer_1") >> ReLU(0.1f) >>
    Linear(32, name = "Layer_2") >> ReLU(0.1f) >>
    Linear(10, name = "OutputLayer")
  val trainingInputLayer = Cast(INT64)
  var loss = SparseSoftmaxCrossEntropy() >> Mean()
  val optimizer = GradientDescent(1e-6)
  val model = Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

  loss = loss >> tf.learn.ScalarSummary("Loss")                  // Collect loss summaries for plotting
  val summariesDir = java.nio.file.Paths.get((tempdir/"summaries").toString())                 // Directory in which to save summaries and checkpoints
  val estimator = Estimator(model, Configuration(Some(summariesDir)))
  estimator.train(
    trainData, StopCriteria(maxSteps = Some(17000)),
    Seq(
      SummarySaverHook(summariesDir, StepHookTrigger(100)),      // Save summaries every 1000 steps
      CheckpointSaverHook(summariesDir, StepHookTrigger(1000))), // Save checkpoint every 1000 steps
    tensorBoardConfig = TensorBoardConfig(summariesDir))         // Launch TensorBoard server in the background
}
