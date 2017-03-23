package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.neuralnets.{Activation, NeuralStack, NeuralStackFactory}
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe, StreamDataPipe, StreamMapPipe}
import org.apache.log4j.Logger
import spire.implicits.cfor

/**
  * @author mandar2812 date 23/03/2017.
  *
  * Generic implementation of the gradient based
  * back-propagation algorithm for training feed forward
  * neural networks.
  *
  * @tparam LayerP The type of the parameters for each layer
  * @tparam I The type of input/output patterns.
  * */
abstract class GradBasedBackPropagation[LayerP, I] extends
  RegularizedOptimizer[NeuralStack[LayerP, I], I, I, Stream[(I, I)]] {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * A data pipeline which takes as input a [[Stream]] of
    * [[Tuple2]] whose first element is the activation and second element
    * the delta value and outputs the gradient of the layer parameters.
    * */
  val gradCompute: DataPipe[Stream[(I, I)], LayerP]

  /**
    * A meta pipeline which for a particular value of the layer parameters,
    * returns a data pipe which takes as input [[Stream]] of [[Tuple2]]
    * consisting of delta's and gradients of activation function with respect to
    * their local fields (calculated via [[Activation.grad]]).
    * */
  val backPropagate: MetaPipe[LayerP, Stream[(I, I)], Stream[I]]

  /**
    * A data pipeline which takes [[Tuple3]] consisting of
    * output layer activations, targets and gradients of output activations
    * with respect to their local fields, respectively and returns
    * the output layer delta values.
    * */
  val computeOutputDelta: StreamMapPipe[(I, I, I), I]

  /**
    * Performs the actual update to the layer parameters
    * after all the gradients have been calculated.
    * */
  val updater: BasicUpdater[Seq[LayerP]]

  val stackFactory: NeuralStackFactory[LayerP, I]

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(
    nPoints: Long, data: Stream[(I, I)],
    initialStack: NeuralStack[LayerP, I]) = {

    //Initialize a working solution to the loss function optimization problem
    var workingStack = initialStack

    val (patterns, targets): (Stream[I], Stream[I]) = data.unzip

    logger.info("------------ Starting back-propagation procedure ------------")

    cfor(1)(count => count < numIterations, count => count + 1)( count => {

      logger.info("------------ Epoch "+count+" ------------")

      /*
      * In each epoch conduct three stages:
      *   1. Forward propagation of fields and activations
      *   2. Backward propagation of deltas
      *   3. Updating of weights based on deltas and activations
      * */

      /*
      * Stage 1
      * */
      logger.info("             Forward propagation ------------")
      val layers = workingStack._layers

      /*
      * Calculate/forward propagate the fields and activations from the input to the output layers
      * [(f0, a0), (f1, a1), ..., (fL, aL)]
      * where (f0, a0) are (mini Batch Features, mini Batch Features)
      * */
      val fieldsAndActivations = layers.scanLeft((patterns, patterns))(
        (inputBatch, layer) => (layer.localField(inputBatch._2), layer.forward(inputBatch._2))
      )

      //Get the output layer activation function
      val outputLayerActFunc = layers.last.activationFunc

      //Get the output layer fields and activations

      //[f0, f1, ..., fL], [a0, a1, ..., al]
      val (fields, activations) = fieldsAndActivations.unzip
      //[fL], [aL]
      val (outputFields, outputActivations) = fieldsAndActivations.last

      /*
      * Stage 2
      * */
      logger.info("             Back propagation")
      //Calculate the gradients of the output layer activations with respect to their local fields.
      val outputActGrads = outputFields.map(outputLayerActFunc.grad(_))
      //Calculate the delta variable for the output layer
      val outputLayerDelta: Stream[I] = computeOutputDelta(
        outputActivations.zip(targets).zip(outputActGrads).map(t => (t._1._1, t._1._2, t._2))
      )

      /*
      * Calculate the deltas for each layer
      * and discard the delta value produced
      * for the input layer.
      * */
      //[d0 | d1, d2, ..., dL]
      val deltasByLayer = layers.zip(fields.init)
        .scanRight(outputLayerDelta)(
          (layer_and_fields, deltas) => {

            val (layer, local_fields) = layer_and_fields
            backPropagate(layer.parameters)(deltas.zip(layer.activationFunc.grad(local_fields)))
          }
        ).tail

      logger.info("             Calculating gradients")
      /*
      * Calculate the gradients for each layer
      * grad_i needs delta_i, a_[i-1]
      * [(a0, delta1), (a1, delta2), (a_[L-1], delta_L)]
      * */
      val gradientsByLayer = activations.init.zip(deltasByLayer).map(c => gradCompute(c._1.zip(c._2)))

      /*
      * Stage 3
      * */
      val new_layer_params = updater.compute(
        workingStack.layerParameters, gradientsByLayer,
        stepSize, count, regParam)._1

      //Spawn the updated network.
      logger.info("             Updating Network Parameters")
      workingStack = stackFactory(new_layer_params)
    })
    //Return the working solution
    workingStack
  }
}

class FFBackProp(
  stackF: NeuralStackFactory[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]]) extends
  GradBasedBackPropagation[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]] {

  type PatternType = DenseVector[Double]

  /**
    * A data pipeline which takes as input a [[Stream]] of
    * [[Tuple2]] whose first element is the activation and second element
    * the delta value and outputs the gradient of the layer parameters.
    **/
  override val gradCompute = StreamDataPipe(
    (c: (PatternType, PatternType)) => (c._2*c._1.t, c._2)) >
    DataPipe((g: Stream[(DenseMatrix[Double], PatternType)]) => {
      val N = g.length.toDouble
      g.map(c => (c._1/N, c._2/N)).reduce((x, y) => (x._1+y._1, x._2+y._2))
    })

  /**
    * A meta pipeline which for a particular value of the layer parameters,
    * returns a data pipe which takes as input [[Stream]] of [[Tuple2]]
    * consisting of delta's and gradients of activation function with respect to
    * their local fields (calculated via [[Activation.grad]]).
    **/
  override val backPropagate = MetaPipe(
    (p: (DenseMatrix[Double], DenseVector[Double])) => (s: Stream[(PatternType, PatternType)]) => {
      s.map(c => (p._1.t*c._1):*c._2)
    })
  /**
    * A data pipeline which takes [[Tuple3]] consisting of
    * output layer activations, targets and gradients of output activations
    * with respect to their local fields, respectively and returns
    * the output layer delta values.
    **/
  override val computeOutputDelta = StreamDataPipe((s: (PatternType, PatternType, PatternType)) => {
    s._3:*(s._1 - s._2)
  })

  /**
    * Performs the actual update to the layer parameters
    * after all the gradients have been calculated.
    **/
  override val updater = new FFLayerUpdater

  override val stackFactory = stackF
}
