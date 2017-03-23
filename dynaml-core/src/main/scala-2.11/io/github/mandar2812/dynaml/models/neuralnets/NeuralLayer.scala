package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}

/**
  * The basic building block of a neural computation stack.
  *
  * @tparam P The type of the parameters/connections of the layer.
  * @tparam I The type of the input supplied to the layer
  * @tparam J The type of the output.
  * */
trait NeuralLayer[P, I, J] {

  /**
    * The layer synapses or connection weights
    * */
  val parameters: P

  val localField: DataPipe[I, J]

  /**
    * Activation function
    * */
  val activationFunc: Activation[J]

  /**
    * Compute the forward pass through the layer.
    * */
  val forward: DataPipe[I, J]

}

object NeuralLayer {

  /**
    * Create a neural computation layer i.e. [[NeuralLayer]] instance
    *
    * @tparam P The type of the layer parameters/weights/connections
    * @tparam I The type of input accepted by the computation layer
    * @param compute Represents the actual computation as a [[MetaPipe]]
    *                i.e. a data pipeline which takes as input a parameter value
    *                and outputs a pipe which represents the layer computation
    * @param activation The activation function
    * */
  def apply[P, I, J](compute: MetaPipe[P, I, J], activation: Activation[J])(params: P) =
    new NeuralLayer[P, I, J] {
      override val parameters = params
      override val activationFunc = activation
      override val localField = compute(parameters)
      /**
        * Compute the forward pass through the layer.
        **/
      override val forward = localField > activationFunc
    }

}

abstract class Vec2VecLayer extends NeuralLayer[
  (DenseMatrix[Double], DenseVector[Double]),
  DenseVector[Double], DenseVector[Double]] {

  val inputDimensions: Int
  val outputDimensions: Int

}

/**
  * A mechanism to generate neural computation layers on the fly.
  * */
class NeuralLayerFactory[P, I, J](
  metaLocalField: MetaPipe[P, I, J],
  val activationFunc: Activation[J]) extends
  DataPipe[P, NeuralLayer[P, I, J]] {

  override def run(params: P) = NeuralLayer(metaLocalField, activationFunc)(params)
}

class Vec2VecLayerFactory(act: Activation[DenseVector[Double]])(inDim: Int, outDim: Int)
  extends NeuralLayerFactory[
    (DenseMatrix[Double], DenseVector[Double]),
    DenseVector[Double], DenseVector[Double]](
    MetaPipe((p: (DenseMatrix[Double], DenseVector[Double])) => (x: DenseVector[Double]) => p._1*x + p._2),
    act) {
  override def run(params: (DenseMatrix[Double], DenseVector[Double])) = {
    require(
      params._1.cols == inDim && params._1.rows == outDim && params._2.length == outDim,
      "Weight matrix and bias vector sizes must be consistent for a Vector to Vector layer")
    super.run(params)
  }
}

