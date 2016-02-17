package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.models.ParameterizedLearner

/**
  *
  * Top level trait defining
  * properties of a neural network.
  *
  * @tparam G The type of the data structure containing the
  *           training data set.
  *
  * @tparam P The underlying base graph from which the [[NeuralGraph]]
  *           object is constructed.
  *
  * @tparam T A subclass of the [[NeuralGraph]] object with [[P]] as the
  *           base graph
  *
  * @tparam Pattern The type of an individual data pattern
  * */
trait NeuralNetwork[G, P, T <: NeuralGraph[P], Pattern] extends
ParameterizedLearner[G, Int, T,
  DenseVector[Double], DenseVector[Double],
  Stream[Pattern]] {

  val inputDimensions: Int

  val outputDimensions: Int

  val hiddenLayers: Int

  val activations: List[(Double) => Double]

  val neuronCounts: List[Int]

  /**
    * Convert the data structure from type [[G]]
    * to a [[Stream]] of [[Pattern]] objects
    *
    * */
  def dataAsStream(d: G): Stream[Pattern]
}