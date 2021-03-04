import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.UByte
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.data.image.CIFARLoader
//import org.platanios.tensorflow.examples

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
object CIFAR {
  private val logger = Logger(LoggerFactory.getLogger("Examples / CIFAR"))

  //implicit val evOutputStructureFloatLong     : OutputStructure[(Output[Float], Output[Long])] = OutputStructure[(Output[Float], Output[Long])]
  //implicit val evOutputToDataTypeFloatLong     : OutputToDataType.Aux[(Output[Float], Output[Long]), (DataType[Float], DataType[Long])] = OutputToDataType[(Output[Float], Output[Long])]
  //implicit val evOutputToShapeFloatLong     : OutputToShape.Aux[(Output[Float], Output[Long]), (Shape, Shape)] = OutputToShape[(Output[Float], Output[Long])]

  // Implicit helpers for Scala 2.11.
  //implicit val evOutputStructureFloatLong : OutputStructure[(Output[Float], Output[Long])]  = evOutputStructureFloatLong
  //implicit val evOutputToDataTypeFloatLong: OutputToDataType[(Output[Float], Output[Long])] = evOutputToDataTypeFloatLong
  //implicit val evOutputToShapeFloatLong   : OutputToShape[(Output[Float], Output[Long])]    = evOutputToShapeFloatLong

  def main(args: Array[String]): Unit = {
    val tempdir = home / "tmp"
    val dataSet =
        CIFARLoader.load(Paths.get(tempdir.toString()), CIFARLoader.CIFAR_10)

    val trainImages = () => tf.data.datasetFromTensorSlices(dataSet.trainImages).map(_.toFloat)
    val trainLabels = () => tf.data.datasetFromTensorSlices(dataSet.trainLabels).map(_.toLong)
    val trainData = () =>
      trainImages().zip(trainLabels())
          .repeat()
          .shuffle(10000)
          .batch(64)
          .prefetch(10)

    logger.info("Building the logistic regression model.")
    val input = tf.learn.Input(FLOAT32, Shape(-1, dataSet.trainImages.shape(1), dataSet.trainImages.shape(2), dataSet.trainImages.shape(3)))
    val trainInput = tf.learn.Input(INT64, Shape(-1))
    val layer = tf.learn.Conv2D[Float]("Layer_0/Conv2D", Shape(2, 2, 3, 16), 1, 1, SameConvPadding) >>
        tf.learn.AddBias[Float]("Layer_0/Bias") >>
        tf.learn.ReLU[Float]("Layer_0/ReLU", 0.1f) >>
        tf.learn.MaxPool[Float]("Layer_0/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        tf.learn.Conv2D[Float]("Layer_1/Conv2D", Shape(2, 2, 16, 32), 1, 1, SameConvPadding) >>
        tf.learn.AddBias[Float]("Bias_1") >>
        tf.learn.ReLU[Float]("Layer_1/ReLU", 0.1f) >>
        tf.learn.MaxPool[Float]("Layer_1/MaxPool", Seq(1, 2, 2, 1), 1, 1, SameConvPadding) >>
        tf.learn.Flatten[Float]("Layer_2/Flatten") >>
        tf.learn.Linear[Float]("Layer_2/Linear", 256) >>
        tf.learn.ReLU[Float]("Layer_2/ReLU", 0.1f) >>
        tf.learn.Linear[Float]("OutputLayer/Linear", 10)
    val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss/CrossEntropy") >>
        tf.learn.Mean[Float]("Loss/Mean") >>
        tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")
    val optimizer = tf.train.AdaGrad(0.1f)

    val model = tf.learn.Model.simpleSupervised(
      input = input,
      trainInput = trainInput,
      layer = layer,
      loss = loss,
      optimizer = optimizer)

    logger.info("Training the linear regression model.")
    val summariesDir = tempdir / "cifar_summaries"
    val estimator = tf.learn.FileBasedEstimator(
      model,
      tf.learn.Configuration(Some(summariesDir.toNIO)),
      tf.learn.StopCriteria(maxSteps = Some(100000)),
      Set(
        tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir.toNIO, trigger = tf.learn.StepHookTrigger(100)),
        tf.learn.CheckpointSaver(summariesDir.toNIO, tf.learn.StepHookTrigger(1000))),
      tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir.toNIO, reloadInterval = 1))
    estimator.train(trainData, tf.learn.StopCriteria(maxSteps = Some(1000)))

    def accuracy(images: Tensor[UByte], labels: Tensor[UByte]): Float = {
      val predictions = estimator.infer(() => images.toFloat)
      predictions
          .argmax(1).toUByte
          .equal(labels).toFloat
          .mean().scalar
    }

    logger.info(s"Train accuracy = ${accuracy(dataSet.trainImages, dataSet.trainLabels)}")
    logger.info(s"Test accuracy = ${accuracy(dataSet.testImages, dataSet.testLabels)}")
  }
}