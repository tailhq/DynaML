{
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.learn.layers.rnn.RNN
  import org.platanios.tensorflow.api.learn.layers.rnn.cell.{BasicLSTMCell, LSTMTuple}
  import java.nio.file.Paths

  import org.platanios.tensorflow.data.text.PTBLoader

  import ammonite.ops.home

  val tempdir = home/"tmp"

  val batchSize   : Int = 20
  val numSteps    : Int = 20
  val prefetchSize: Int = 10

  val dataType              : DataType = FLOAT32
  val vocabularySize        : Int      = 10000
  val numHidden             : Int      = 200
  val numLayers             : Int      = 1
  val dropoutKeepProbability: Float    = 0.5f

  object RNNOutputLayer extends tf.learn.Layer[LSTMTuple, Output]("RNNOutputLayer") {
    override val layerType: String = "RNNOutputLayer"

    override def forward(input: LSTMTuple, mode: tf.learn.Mode): tf.learn.LayerInstance[LSTMTuple, Output] = {

      val weights = variable("OutputWeights", dataType, Shape(numHidden, vocabularySize))
      val bias = variable("OutputBias", dataType, Shape(vocabularySize))
      val output = tf.linear(tf.reshape(input.output, Shape(-1, numHidden)), weights.value, bias.value)
      // We reshape the output logits to feed into the sequence loss layer
      val reshapedOutput = tf.reshape(output, Shape(batchSize, numSteps, vocabularySize))
      tf.learn.LayerInstance(input, reshapedOutput, trainableVariables = Set(weights, bias))
    }
  }

  val model = {

    val input = tf.learn.Input(INT32, Shape(batchSize, numSteps))
    val trainInput = tf.learn.Input(INT32, Shape(batchSize, numSteps))

    // Slightly better results can be obtained with forget gate biases initialized to 1 but the hyper-parameters of the
    // model would need to be different than those reported in the paper.

    val rnnCell = BasicLSTMCell(numHidden, FLOAT32, Shape(-1, numHidden), forgetBias = 0.0f)

    // TODO: Add dropout wrapper.
    // TODO: Add multi-RNN cell.
    val rnn = RNN(rnnCell, timeMajor = false)

    val layer = tf.learn.device("/device:CPU:0") {
      tf.learn.Embedding(vocabularySize, numHidden, dataType)
    } >> tf.learn.Dropout(dropoutKeepProbability) >> rnn >> RNNOutputLayer

    val loss = tf.learn.SequenceLoss(averageAcrossTimeSteps = false, averageAcrossBatch = true) >>
      tf.learn.Sum() >>
      tf.learn.ScalarSummary("Loss")

    val optimizer = tf.train.GradientDescent(1.0)

    tf.learn.Model(input, layer, trainInput, loss, optimizer, tf.learn.ClipGradientsByGlobalNorm(5.0f))
  }


  val dataset = PTBLoader.load(Paths.get((tempdir/"PTB").toString()))
  val trainDataset =
    PTBLoader.tokensToBatchedTFDataset(dataset.train, batchSize, numSteps, "TrainDataset")
      .repeat()
      .prefetch(prefetchSize)

  val summariesDir = Paths.get((tempdir/"rnn-ptb").toString())

  val estimator = tf.learn.InMemoryEstimator(
    model,
    tf.learn.Configuration(Some(summariesDir)),
    tf.learn.StopCriteria(maxSteps = Some(100000)),
    Set(
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = tf.learn.StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
      tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(100))),
    tensorBoardConfig = tf.learn.TensorBoardConfig(logDir = summariesDir, reloadInterval = 100))

  estimator.train(() => trainDataset, tf.learn.StopCriteria(maxSteps = Some(500)))
}
