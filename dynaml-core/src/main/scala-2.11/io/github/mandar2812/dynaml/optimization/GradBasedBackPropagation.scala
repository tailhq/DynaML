package io.github.mandar2812.dynaml.optimization

import io.github.mandar2812.dynaml.models.neuralnets.{NeuralStack, NeuralStackFactory}
import io.github.mandar2812.dynaml.pipes.{MetaPipe, StreamDataPipe, StreamMapPipe}
import org.apache.log4j.Logger
import spire.implicits.cfor

/**
  * Created by mandar2812 on 23/03/2017.
  */
abstract class GradBasedBackPropagation[LayerP, I] extends
  RegularizedOptimizer[NeuralStack[LayerP, I], I, I, Stream[(I, I)]] {

  private val logger = Logger.getLogger(this.getClass)

  val gradCompute: StreamDataPipe[(I, I), LayerP, LayerP]

  val backPropagate: MetaPipe[LayerP, Stream[(I, I)], Stream[I]]

  val computeOutputDelta: StreamMapPipe[(I, I, I), I]

  val updater: BasicUpdater[Seq[LayerP]]

  val stackFactory: NeuralStackFactory[LayerP, I]

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long, data: Stream[(I, I)], initialStack: NeuralStack[LayerP, I]) = {
    var workingStack = initialStack

    val (patterns, targets): (Stream[I], Stream[I]) = data.unzip

    logger.info("------------ Starting Back-propagation procedure ------------")

    cfor(1)(count => count < numIterations, count => count + 1)( count => {

      logger.info("------------ Epoch "+count+"------------")

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
      * [(f0, a0), (f1, a1), ...]
      * where (f0, a0) are (mini Batch Features, mini Batch Features)
      * */
      val fieldsAndActivations = layers.scanLeft((patterns, patterns))(
        (inputBatch, layer) => (layer.localField(inputBatch._2), layer.forward(inputBatch._2))
      )

      //Get the output layer activation function
      val outputLayerActFunc = layers.last.activationFunc

      //Get the output layer fields and activations
      val (fields, activations) = fieldsAndActivations.unzip
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

      //Calculate the deltas for each layer
      val deltasByLayer = layers.zip(fields.tail).scanRight(outputLayerDelta)((layer_and_fields, deltas) => {
        val (layer, local_fields) = layer_and_fields

        backPropagate(layer.parameters)(deltas.zip(layer.activationFunc.grad(local_fields)))
      }).tail

      val gradientsByLayer = activations.init.zip(deltasByLayer).map(c => gradCompute(c._1.zip(c._2)))

      val new_layer_params = updater.compute(
        workingStack.layerParameters, gradientsByLayer,
        stepSize, count, regParam)._1

      logger.info("------------ Updating Network Parameters ------------")
      workingStack = stackFactory(new_layer_params)
    })

    workingStack
  }
}
