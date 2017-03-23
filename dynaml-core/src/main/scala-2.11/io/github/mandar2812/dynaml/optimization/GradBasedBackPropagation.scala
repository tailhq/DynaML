package io.github.mandar2812.dynaml.optimization

import io.github.mandar2812.dynaml.models.neuralnets.{NeuralStack, NeuralStackFactory, Activation}
import io.github.mandar2812.dynaml.pipes.{MetaPipe, StreamDataPipe, StreamMapPipe}
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
  val gradCompute: StreamDataPipe[(I, I), LayerP, LayerP]

  /**
    * A meta pipeline which for a particular value of the layer parameters,
    * returns a data pipe which takes as input [[Stream]] of [[Tuple2]]
    * consisting of delta's and gradients of activation function with respect to
    * their local fields (calculated via [[Activation.grad]]).
    * */
  val backPropagate: MetaPipe[LayerP, Stream[(I, I)], Stream[I]]

  /**
    * A data pipeline which takes [[Tuple3]] consisting of
    * output layer activations, targets and
    * gradients of output activations
    * with respect to their local fields,
    * respectively and outputs the output layer
    * delta values.
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

    logger.info("------------ Starting Back-propagation procedure ------------")

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
      val deltasByLayer = layers.zip(fields.tail)
        .scanRight(outputLayerDelta)(
          (layer_and_fields, deltas) => {

            val (layer, local_fields) = layer_and_fields
            backPropagate(layer.parameters)(deltas.zip(layer.activationFunc.grad(local_fields)))
          }
        ).tail

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
      logger.info("------------ Updating Network Parameters ------------")
      workingStack = stackFactory(new_layer_params)
    })
    //Return the working solution
    workingStack
  }
}
