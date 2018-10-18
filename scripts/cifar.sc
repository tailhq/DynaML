{
  import ammonite.ops._
  import io.github.mandar2812.dynaml.pipes.DataPipe
  import io.github.mandar2812.dynaml.tensorflow.data.AbstractDataSet
  import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
  import io.github.mandar2812.dynaml.tensorflow.implicits._
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.learn.layers.Activation
  import org.platanios.tensorflow.api.learn.layers.Layer
  import org.platanios.tensorflow.api.ops.NN.SameConvPadding
  import org.platanios.tensorflow.api.learn.layers.Input
  import org.platanios.tensorflow.api.types.MathDataType
  import org.platanios.tensorflow.data.image.CIFARLoader
  import java.nio.file.Paths


  val tempdir = home/"tmp"

  val dataSet = CIFARLoader.load(Paths.get(tempdir.toString()), CIFARLoader.CIFAR_10)
  val tf_dataset = AbstractDataSet[Tensor[DataType], Tensor[DataType]](
    dataSet.trainImages, dataSet.trainLabels, dataSet.trainLabels.shape(0),
    dataSet.testImages, dataSet.testLabels, dataSet.testLabels.shape(0))

  val trainData =
    tf_dataset.training_data
      .repeat()
      .shuffle(10000)
      .batch(128)
      .prefetch(10)


  println("Building the model.")
  val input: Input[Tensor[DataType], Output, DataType, Shape] = tf.learn.Input(
    UINT8,
    Shape(-1) ++ dataSet.trainImages.shape(1::)
  )

  val trainInput: Input[Tensor[DataType], Output, DataType, Shape] = tf.learn.Input(UINT8, Shape(-1))

  val relu_act = DataPipe[String, Activation](tf.learn.ReLU(_))

  val architecture = tf.learn.Cast("Input/Cast", FLOAT32) >>
    dtflearn.inception_unit(channels = 3,  Seq.fill(4)(10), relu_act)(layer_index = 1) >>
    dtflearn.inception_unit(channels = 40, Seq.fill(4)(5),  relu_act)(layer_index = 2) >>
    tf.learn.Flatten("Layer_3/Flatten") >>
    dtflearn.feedforward(256)(id = 4) >>
    tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
    dtflearn.feedforward(10)(id = 5)

  val trainingInputLayer = tf.learn.Cast("TrainInput/Cast", INT64)

  val loss = tf.learn.SparseSoftmaxCrossEntropy("Loss/CrossEntropy") >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.Adam(0.1f)

  val summariesDir = java.nio.file.Paths.get((tempdir/"cifar_summaries").toString())

  val (model, estimator) = dtflearn.build_tf_model(
    architecture, input, trainInput, trainingInputLayer,
    loss, optimizer, summariesDir, dtflearn.max_iter_stop(500),
    100, 100, 100)(
    trainData, false)

  def accuracy[D <: MathDataType](predictions: Tensor[D], labels: Tensor[D]): Float =
    tfi.equal(predictions.argmax(1).cast(UINT8), labels.cast(UINT8))
      .cast(FLOAT32)
      .mean()
      .scalar
      .asInstanceOf[Float]

  val (trainingPreds, testPreds): (Option[Tensor[DataType]], Option[Tensor[DataType]]) =
    dtfutils.predict_data[
      Tensor[DataType], Output, DataType, Shape, Output,
      Tensor[DataType], Output, DataType, Shape, Output,
      Tensor[DataType], Tensor[DataType]](
      estimator, data = tf_dataset, pred_flags = (true, true), buff_size = 20000)

  val (trainAccuracy, testAccuracy) = (
    accuracy(trainingPreds.get.cast(UINT8), dataSet.trainLabels),
    accuracy(testPreds.get.cast(UINT8), dataSet.testLabels))

  print("Train accuracy = ")
  pprint.pprintln(trainAccuracy)

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)
}
