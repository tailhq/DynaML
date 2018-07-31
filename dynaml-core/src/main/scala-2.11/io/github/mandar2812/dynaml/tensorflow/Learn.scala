package io.github.mandar2812.dynaml.tensorflow

import io.github.mandar2812.dynaml.tensorflow.layers.{DynamicTimeStepCTRNN, FiniteHorizonCTRNN, FiniteHorizonLinear}
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.learn.layers.{Activation, Input, Layer}
import org.platanios.tensorflow.api.ops.NN.SameConvPadding
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.io.data.Dataset
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.{FLOAT32, Graph, Shape, Tensor, tf, _}

private[tensorflow] object Learn {

  type TFDATA = Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)]

  val Phi: layers.Phi.type                           = layers.Phi
  val Tanh: layers.Tanh.type                         = layers.Tanh
  val GeneralizedLogistic
  : layers.GeneralizedLogistic.type                  = layers.GeneralizedLogistic

  val batch_norm: layers.BatchNormalisation.type     = layers.BatchNormalisation
  val ctrnn: layers.FiniteHorizonCTRNN.type          = layers.FiniteHorizonCTRNN
  val dctrnn: layers.DynamicTimeStepCTRNN.type       = layers.DynamicTimeStepCTRNN
  val ts_linear: layers.FiniteHorizonLinear.type     = layers.FiniteHorizonLinear
  val rbf_layer: layers.RBFLayer.type                = layers.RBFLayer
  val stack_outputs: layers.StackOutputs.type        = layers.StackOutputs
  val concat_outputs: layers.ConcatenateOutputs.type = layers.ConcatenateOutputs
  val seq_layer: layers.SeqLayer.type                = layers.SeqLayer
  val combined_layer: layers.CombinedLayer.type      = layers.CombinedLayer
  val unstack: layers.Unstack.type                   = layers.Unstack
  val identity: layers.IdentityLayer.type            = layers.IdentityLayer
  val tuple2_layer: layers.Tuple2Layer.type          = layers.Tuple2Layer
  val stack_tuple2: layers.StackTuple2.type          = layers.StackTuple2
  val concat_tuple2: layers.ConcatenateTuple2.type   = layers.ConcatenateTuple2

  /**
    * Stop after a specified maximum number of iterations has been reached.
    * */
  val max_iter_stop: Long => StopCriteria           = (n: Long) => tf.learn.StopCriteria(maxSteps = Some(n))

  /**
    * Stop after the change in the loss function falls below a specified threshold.
    * */
  val abs_loss_change_stop: (Double, Long) => StopCriteria  = (d: Double, max_iter: Long) => tf.learn.StopCriteria(
    absLossChangeTol = Some(d),
    maxSteps = Some(max_iter))

  /**
    * Stop after the relative change in the loss function falls below a specified threshold.
    * */
  val rel_loss_change_stop: (Double, Long) => StopCriteria  = (d: Double, max_iter: Long) => tf.learn.StopCriteria(
    relLossChangeTol = Some(d),
    maxSteps = Some(max_iter))

  /**
    * Constructs a feed-forward layer.
    *
    * @param num_units The number of neurons in the layer
    * @param id A unique integer id for constructing the layer name.
    *
    * */
  def feedforward(num_units: Int, useBias: Boolean = true)(id: Int) =
    tf.learn.Linear("Linear_"+id, num_units, useBias)

  /**
    * Constructs a simple feed-forward stack of layers.
    *
    * @param get_act A function which given a layer index number,
    *                returns an activation function.
    *
    * @param dataType The data type of the layer weights/biases.
    *
    * @param layer_sizes A Sequence of layer sizes/dimensions/neuron counts.
    *
    * @param starting_index Specify which layer number should the indexing of
    *                       the layers start with, defaults to 1.
    * */
  def feedforward_stack(
    get_act: Int => Activation,
    dataType: DataType)(
    layer_sizes: Seq[Int],
    starting_index: Int = 1): Layer[Output, Output] = {

    def stack_ff_layers_rec(
      ls: Seq[Int],
      layer_acc: Layer[Output, Output],
      layer_index: Int): Layer[Output, Output] = ls match {

      case Seq() => layer_acc

      case Seq(num_output_units) => layer_acc >> dtflearn.feedforward(num_output_units)(layer_index)

      case _ => stack_ff_layers_rec(
        ls.tail,
        layer_acc >> dtflearn.feedforward(ls.head)(layer_index) >> get_act(layer_index),
        layer_index + 1)
    }

    stack_ff_layers_rec(
      layer_sizes, tf.learn.Cast("Input/Cast", dataType),
      starting_index)

  }

  /**
    * Constructs a symmetric (square) convolutional layer from the provided dimensions.
    *
    * [[org.platanios.tensorflow.api.ops.NN.SameConvPadding]] is used as the padding mode.
    *
    * @param size The size of each square filter e.g. 2*2, 3*3 etc
    * @param num_channels_input The number of channels in the input
    * @param num_filters The number of channels in the layer output
    * @param strides A [[Tuple2]] with strides, for each direction i.e. breadth and height.
    * @param index The layer id or index, helps in creating a unique layer name
    * */
  def conv2d(size: Int, num_channels_input: Int, num_filters: Int, strides: (Int, Int))(index: Int) =
    tf.learn.Conv2D(
      "Conv2D_"+index,
      Shape(size, size, num_channels_input, num_filters),
      strides._1, strides._2,
      SameConvPadding)

  /**
    * Constructs a convolutional layer activated by a ReLU, with
    * an option of appending a dropout layer.
    *
    * */
  def conv2d_unit(
    shape: Shape, stride: (Int, Int) = (1, 1),
    relu_param: Float = 0.1f, dropout: Boolean = true,
    keep_prob: Float = 0.6f)(i: Int) =
    if(dropout) {
      tf.learn.Conv2D("Conv2D_"+i, shape, stride._1, stride._2, SameConvPadding) >>
        tf.learn.AddBias(name = "Bias_"+i) >>
        tf.learn.ReLU("ReLU_"+i, relu_param) >>
        tf.learn.Dropout("Dropout_"+i, keep_prob)
    } else {
      tf.learn.Conv2D("Conv2D_"+i, shape, stride._1, stride._2, SameConvPadding) >>
        batch_norm(name = "BatchNorm_"+i) >>
        tf.learn.ReLU("ReLU_"+i, relu_param) >>
        tf.learn.Cast("Cast_"+i, FLOAT32)
    }

  /**
    * Constructs an inverted convolutional pyramid, consisting of
    * stacked versions of [Conv2d --> ReLU --> Dropout] layers.
    *
    * The number of filters learned in each Conv2d layer are
    * arranged in decreasing exponents of 2. They are costructed
    * using calls to [[conv2d_unit()]]
    *
    * ... Conv_unit(128) --> Conv_unit(64) --> Conv_unit(32) --> Conv_unit(16) ...
    *
    * @param size The size of the square convolutional filter to be applied
    *             in each segment.
    * @param num_channels_input The number of channels in the input.
    * @param start_num_bits The exponent of 2 which determines size/depth of the starting layer
    *                       e.g. set to 4 for a depth of 16.
    *
    * @param end_num_bits The exponent of 2 which determines the size/depth of the end layer.
    *
    * @param relu_param The activation barrier of the ReLU activation.
    *
    * @param dropout Set to true, if dropout layers should be placed in each convolutional unit.
    *                Set to false, and batch normalisation layers shall be placed after each convolutional unit.
    *
    * @param keep_prob If dropout is enabled, then this determines the retain probability.
    * */
  def conv2d_pyramid(
    size: Int, num_channels_input: Int)(
    start_num_bits: Int, end_num_bits: Int)(
    relu_param: Float = 0.1f, dropout: Boolean = true,
    keep_prob: Float = 0.6f, starting_index: Int = 0) = {

    require(
      start_num_bits > end_num_bits,
      "To construct a 2d-convolutional pyramid, you need to start_num_bits > end_num_bits")

    //Create the first layer segment.
    val head_segment = conv2d_unit(
      Shape(size, size, num_channels_input, math.pow(2, start_num_bits).toInt),
      stride = (1, 1), relu_param, dropout, keep_prob)(starting_index)

    //Create the rest of the pyramid
    val tail_segments = (end_num_bits until start_num_bits).reverse.zipWithIndex.map(bitsAndIndices => {
      val (bits, index) = bitsAndIndices

      conv2d_unit(
        Shape(size, size, math.pow(2, bits+1).toInt, math.pow(2, bits).toInt),
        stride = (math.pow(2, index+1).toInt, math.pow(2, index+1).toInt),
        relu_param, dropout, keep_prob)(index+1+starting_index)

    }).reduceLeft((a,b) => a >> b)

    //Join head to tail.
    head_segment >> tail_segments
  }

  /**
    * Constructs an Inception v2 architecture
    * computational unit.
    * */
  def inception_unit(
    channels: Int,
    num_filters: Seq[Int],
    relu_param: Float = 0.01f)(
    layer_index: Int): Layer[Output, Output] = {

    require(num_filters.length == 4,
      s"Inception module has only 4 branches, but ${num_filters.length}" +
        s" were assumed while setting num_filters variable")

    val name = s"Inception_$layer_index"

    val branch1 =
      tf.learn.Conv2D(
        s"$name/B1/Conv2D_1x1",
        Shape(1, 1, channels, num_filters.head),
        1, 1, SameConvPadding) >>
        tf.learn.ReLU(s"$name/B1/ReLU_1", relu_param)

    val branch2 =
      tf.learn.Conv2D(s"$name/B2/Conv2D_1x1", Shape(1, 1, channels, num_filters(1)), 1, 1, SameConvPadding) >>
        tf.learn.ReLU(s"$name/B2/ReLU_1", relu_param) >>
        tf.learn.Conv2D(s"$name/B2/Conv2D_1x3", Shape(1, 3, num_filters(1), num_filters(1)), 1, 1, SameConvPadding) >>
        tf.learn.Conv2D(s"$name/B2/Conv2D_3x1", Shape(3, 1, num_filters(1), num_filters(1)), 1, 1, SameConvPadding) >>
        tf.learn.ReLU(s"$name/B2/ReLU_2", relu_param)

    val branch3 =
      tf.learn.Conv2D(s"$name/B3/Conv2D_1x1", Shape(1, 1, channels, num_filters(2)), 1, 1, SameConvPadding) >>
      tf.learn.ReLU(s"$name/B3/ReLU_1", relu_param) >>
      tf.learn.Conv2D(s"$name/B3/Conv2D_1x3_1", Shape(1, 3, num_filters(2), num_filters(2)), 1, 1, SameConvPadding) >>
      tf.learn.Conv2D(s"$name/B3/Conv2D_3x1_1", Shape(3, 1, num_filters(2), num_filters(2)), 1, 1, SameConvPadding) >>
      tf.learn.Conv2D(s"$name/B3/Conv2D_1x3_2", Shape(1, 3, num_filters(2), num_filters(2)), 1, 1, SameConvPadding) >>
      tf.learn.Conv2D(s"$name/B3/Conv2D_3x1_2", Shape(3, 1, num_filters(2), num_filters(2)), 1, 1, SameConvPadding) >>
      tf.learn.ReLU(s"$name/B3/ReLU_2", relu_param)

    val branch4 = tf.learn.MaxPool(s"$name/B4/MaxPool", Seq(1, 3, 3, 1), 1, 1, SameConvPadding) >>
      tf.learn.Conv2D(s"$name/B4/Conv2D_1x1", Shape(1, 1, channels, num_filters(3)), 1, 1, SameConvPadding) >>
      tf.learn.ReLU(s"$name/B4/ReLU_1", relu_param)

    val layers = Seq(
      branch1, branch2, branch3, branch4
    )

    combined_layer(name, layers) >> concat_outputs(name+"/DepthConcat", -1)

  }

  /**
    * Constructs a Continuous Time Recurrent Neural Network (CTRNN) Layer, consisting
    * of some latent states, composed with a linear projection into the space of observables.
    *
    * @param observables The dimensionality of the output space.
    * @param timestep The integration time step, if set to 0 or a negative
    *                 value, create a [[DynamicTimeStepCTRNN]].
    * @param horizon The number of steps in time to simulate the dynamical system
    * @param index The layer index, should be unique.
    * */
  def ctrnn_block(
    observables: Int,
    horizon: Int, timestep: Double = -1d)(index: Int) =
    if (timestep <= 0d) {
      DynamicTimeStepCTRNN(s"FHctrnn_$index", horizon) >>
        FiniteHorizonLinear(s"FHlinear_$index", observables)
    } else {
      FiniteHorizonCTRNN(s"FHctrnn_$index", horizon, timestep) >>
        FiniteHorizonLinear(s"FHlinear_$index", observables)
    }

  /**
    * Trains a tensorflow model/estimator.
    *
    * @tparam IT The type representing input tensors,
    *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
    *
    * @tparam IO The type representing symbolic tensors of the input patterns,
    *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
    *
    * @tparam IDA The underlying (scalar) data types of the input,
    *             e.g. `DataType.Aux[Double]`, `(DataType.Aux[Double], DataType.Aux[Double])` etc.
    *
    * @tparam ID The input pattern's tensorflow data type,
    *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
    *
    * @tparam IS The type of the input pattern's shape,
    *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
    *
    * @tparam I The type of the symbolic tensor returned by the neural architecture,
    *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
    *
    * @tparam TT The type representing target/label tensors,
    *            e.g. `Tensor`, `(Tensor, Tensor)`, `Seq[Tensor]`  etc.
    * @tparam TO The type representing symbolic tensors of the target patterns,
    *            e.g. `Output`, `(Output, Output)`, `Seq[Output]` etc.
    * @tparam TDA The underlying (scalar) data types of the targets,
    *             e.g. `DataType.Aux[Double]`, `(DataType.Aux[Double], DataType.Aux[Double])` etc.
    *
    * @tparam TD The target pattern's tensorflow data type,
    *            e.g. `FLOAT64`, `(FLOAT64, FLOAT64)`, etc.
    *
    * @tparam TS The type of the target pattern's shape,
    *            e.g. `Shape`, `(Shape, Shape)`, `Seq[Shape]`
    *
    * @tparam T The type of the symbolic tensor of the processed targets, this is the type
    *           of the tensorflow symbol which is used to compute the loss.
    *           e.g. `Output`, `(Output, Output)`, `Seq[Output]`
    *
    * @param architecture The network architecture,
    *                     takes a value of type [[IO]] and returns
    *                     a value of type [[I]].
    * @param input The input meta data.
    * @param target The output label meta data
    * @param processTarget A computation layer which converts
    *                      the original target of type [[TO]]
    *                      into a type [[T]], usable by the Estimator API
    * @param loss The loss function to be optimized during training.
    * @param optimizer The optimization algorithm implementation.
    * @param summariesDir A filesystem path of type [[java.nio.file.Path]], which
    *                     determines where the intermediate model parameters/checkpoints
    *                     will be written.
    * @param stopCriteria The stopping criteria for training, for examples see
    *                     [[max_iter_stop]], [[abs_loss_change_stop]] and [[rel_loss_change_stop]]
    *
    * @param stepRateFreq The frequency at which to log the step rate (expressed as number of iterations/sec).
    * @param summarySaveFreq The frequency at which to log the loss summary.
    * @param checkPointFreq The frequency at which to log the model parameters.
    * @param training_data A training data set, as an instance of [[Dataset]].
    * @param inMemory Set to true if the estimator should be in-memory.
    *
    * @return A [[Tuple2]] containing the model and estimator.
    *
    * @author mandar2812
    * */
  def build_tf_model[
  IT, IO, IDA, ID, IS, I,
  TT, TO, TDA, TD, TS, T](
    architecture: Layer[IO, I],
    input: Input[IT, IO, IDA, ID, IS],
    target: Input[TT, TO, TDA, TD, TS],
    processTarget: Layer[TO, T],
    loss: Layer[(I, T), Output],
    optimizer: Optimizer,
    summariesDir: java.nio.file.Path,
    stopCriteria: StopCriteria,
    stepRateFreq: Int = 5000,
    summarySaveFreq: Int = 5000,
    checkPointFreq: Int = 5000)(
    training_data: Dataset[
      (IT, TT), (IO, TO),
      (ID, TD), (IS, TS)],
    inMemory: Boolean = false
  ) = {

    val (model, estimator) = tf.createWith(graph = Graph()) {
      val model = tf.learn.Model.supervised(
        input, architecture,
        target, processTarget,
        loss, optimizer)

      println("\nTraining model.\n")

      val estimator = if(inMemory) {

        tf.learn.InMemoryEstimator(
          model,
          tf.learn.Configuration(Some(summariesDir)),
          stopCriteria,
          Set(
            tf.learn.StepRateLogger(
              log = false, summaryDir = summariesDir,
              trigger = tf.learn.StepHookTrigger(stepRateFreq)),
            tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(summarySaveFreq)),
            tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(checkPointFreq))),
          tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = checkPointFreq))

      } else {

        tf.learn.FileBasedEstimator(
          model,
          tf.learn.Configuration(Some(summariesDir)),
          stopCriteria,
          Set(
            tf.learn.StepRateLogger(
              log = false, summaryDir = summariesDir,
              trigger = tf.learn.StepHookTrigger(stepRateFreq)),
            tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(summarySaveFreq)),
            tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(checkPointFreq))),
          tensorBoardConfig = tf.learn.TensorBoardConfig(summariesDir, reloadInterval = checkPointFreq))

      }

      estimator.train(() => training_data)

      (model, estimator)
    }

    (model, estimator)
  }

}
