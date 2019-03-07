{
  import ammonite.ops._
  import io.github.mandar2812.dynaml.pipes.DataPipe
  import io.github.mandar2812.dynaml.tensorflow.data.AbstractDataSet
  import io.github.mandar2812.dynaml.tensorflow.{dtflearn, dtfutils}
  import io.github.mandar2812.dynaml.tensorflow.implicits._
  import org.platanios.tensorflow.api._
  //import org.platanios.tensorflow.api.implicits.helpers._
  import org.platanios.tensorflow.api.learn.layers.Activation
  import org.platanios.tensorflow.data.image.CIFARLoader
  import java.nio.file.Paths


  // Implicit helpers for Scala 2.11.
  //implicit val evOutputStructureFloatLong : OutputStructure[(Output[Float], Output[Long])]  = OutputStructure[(Output[Float], Output[Long])]
  //implicit val evOutputToDataTypeFloatLong: OutputToDataType[(Output[Float], Output[Long])] = OutputToDataType[(Output[Float], Output[Long])]
  //implicit val evOutputToShapeFloatLong   : OutputToShape[(Output[Float], Output[Long])]    = OutputToShape[(Output[Float], Output[Long])]


  val tempdir = home/"tmp"

  val dataSet = CIFARLoader.load(Paths.get(tempdir.toString()), CIFARLoader.CIFAR_10)
  val tf_dataset = AbstractDataSet(
    dataSet.trainImages, dataSet.trainLabels.castTo[Long], dataSet.trainLabels.shape(0),
    dataSet.testImages, dataSet.testLabels.castTo[Long], dataSet.testLabels.shape(0))

  val trainData = 
    tf_dataset.training_data
      .repeat()
      .shuffle(10000)
      .batch[(UINT8, INT64), (Shape, Shape)](128)
      .prefetch(10)


  println("Building the model.")
  val input = tf.learn.Input(
    UINT8,
    Shape(-1) ++ dataSet.trainImages.shape(1::)
  )

  val trainInput = tf.learn.Input(INT64, Shape(-1))

  val relu_act = DataPipe[String, Activation[Float]](tf.learn.ReLU(_))

  val architecture =
    tf.learn.Cast[UByte, Float]("Input/Cast") >>
    dtflearn.inception_unit(channels = 3,  Seq.fill(4)(10), relu_act)(layer_index = 1) >>
    dtflearn.inception_unit(channels = 40, Seq.fill(4)(5),  relu_act)(layer_index = 2) >>
    tf.learn.Flatten("Layer_3/Flatten") >>
    dtflearn.feedforward(256)(id = 4) >>
    tf.learn.ReLU("Layer_4/ReLU", 0.1f) >>
    dtflearn.feedforward(10)(id = 5)

  val loss = tf.learn.SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss/CrossEntropy") >>
    tf.learn.Mean("Loss/Mean") >>
    tf.learn.ScalarSummary("Loss/Summary", "Loss")

  val optimizer = tf.train.Adam(0.1f)

  val summariesDir = java.nio.file.Paths.get((tempdir/"cifar_summaries").toString())

  val (model, estimator) = dtflearn.build_tf_model[
    Output[UByte], Output[Long], Output[Float], Output[Float],
    Float, (Output[Float], Output[Long]),
    UINT8, Shape, INT64, Shape](
    architecture, input, trainInput,
    loss, optimizer, summariesDir, dtflearn.max_iter_stop(500),
    100, 100, 100)(trainData, false)

  def accuracy(predictions: Tensor[Long], labels: Tensor[Long]): Float =
    tfi.equal(predictions.argmax[Long](1), labels)
      .castTo[Float]
      .mean()
      .scalar
      .asInstanceOf[Float]

  val (trainingPreds, testPreds): (Option[Tensor[Float]], Option[Tensor[Float]]) =
    dtfutils.predict_data[
      Output[UByte], Output[Long],
      Output[Float], Output[Float], Float,
      Tensor[UByte], UINT8, Shape,
      Tensor[Float], FLOAT32, Shape,
      Tensor[Long], Tensor[Float]](
      predictiveModel = estimator, data = tf_dataset,
      pred_flags = (true, true),
      buff_size = 20000
    )

  val (trainAccuracy, testAccuracy) = (
    accuracy(trainingPreds.get.castTo[Long], dataSet.trainLabels.castTo[Long]),
    accuracy(testPreds.get.castTo[Long], dataSet.testLabels.castTo[Long]))

  print("Train accuracy = ")
  pprint.pprintln(trainAccuracy)

  print("Test accuracy = ")
  pprint.pprintln(testAccuracy)
}
