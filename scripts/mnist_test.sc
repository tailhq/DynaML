import _root_.java.nio.file.Paths
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.data.image.MNISTLoader

val dataSet = MNISTLoader.load(Paths.get("datasets/MNIST"))
val trainImages =
  tf.data.datasetFromOutputSlices(dataSet.trainImages.toOutput).map(_.toFloat)
val trainLabels =
  tf.data.datasetFromOutputSlices(dataSet.trainLabels.toOutput).map(_.toLong)
val testImages =
  tf.data.datasetFromOutputSlices(dataSet.testImages.toOutput).map(_.toFloat)
val testLabels =
  tf.data.datasetFromOutputSlices(dataSet.testLabels.toOutput).map(_.toLong)
val trainData =
  trainImages
    .zip(trainLabels)
    .repeat()
    .shuffle(10000)
    .batch(256)
    .prefetch(10)
val evalTrainData = trainImages.zip(trainLabels).batch(1000).prefetch(10)
val evalTestData  = testImages.zip(testLabels).batch(1000).prefetch(10)

logger.info("Building the logistic regression model.")

val input = tf.learn.Input(
  FLOAT32,
  Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2))
)
val trainInput = tf.learn.Input(INT64, Shape(-1))
val layer = tf.learn.Flatten[Float]("Input/Flatten") >>
  tf.learn.Linear[Float]("Layer_0/Linear", 128) >> tf.learn
  .ReLU[Float]("Layer_0/ReLU", 0.1f) >>
  tf.learn.Linear[Float]("Layer_1/Linear", 64) >> tf.learn
  .ReLU[Float]("Layer_1/ReLU", 0.1f) >>
  tf.learn.Linear[Float]("Layer_2/Linear", 32) >> tf.learn
  .ReLU[Float]("Layer_2/ReLU", 0.1f) >>
  tf.learn.Linear[Float]("OutputLayer/Linear", 10)
val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float](
  "Loss/CrossEntropy"
) >>
  tf.learn.Mean[Float]("Loss/Mean") >>
  tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")
val optimizer = tf.train.YellowFin()

val model = tf.learn.Model.simpleSupervised(
  input = input,
  trainInput = trainInput,
  layer = layer,
  loss = loss,
  optimizer = optimizer,
  clipGradients = tf.learn.ClipGradientsByGlobalNorm(5.0f)
)

logger.info("Training the linear regression model.")
val summariesDir = Paths.get("temp/mnist-mlp")
val accMetric =
  tf.metrics.MapMetric((v: (Output[Float], (Output[Float], Output[Int]))) => {
    (tf.argmax(v._1, -1, INT64).toFloat, v._2._2.toFloat)
  }, tf.metrics.Accuracy("Accuracy"))
val estimator = tf.learn.InMemoryEstimator(
  model,
  tf.learn.Configuration(Some(summariesDir)),
  tf.learn.StopCriteria(maxSteps = Some(100000)),
  Set(
    tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
    tf.learn.Evaluator(
      log = true,
      datasets =
        Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
      metrics = Seq(accMetric),
      trigger = tf.learn.StepHookTrigger(1000),
      name = "Evaluator"
    ),
    tf.learn.StepRateLogger(
      log = false,
      summaryDir = summariesDir,
      trigger = tf.learn.StepHookTrigger(100)
    ),
    tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
    tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))
  ),
  tensorBoardConfig =
    tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
)
estimator.train(
  () => trainData,
  tf.learn.StopCriteria(maxSteps = Some(10000))
)

def accuracy(images: Tensor[UByte], labels: Tensor[UByte]): Float = {
  val predictions = estimator.infer(() => images.toFloat)
  predictions
    .argmax(1)
    .toUByte
    .equal(labels)
    .toFloat
    .mean()
    .scalar
}

logger.info(
  s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}"
)
logger.info(
  s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}"
)
