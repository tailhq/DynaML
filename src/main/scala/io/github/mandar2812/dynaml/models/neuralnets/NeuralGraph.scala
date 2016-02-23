package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector

/**
  * A Neural Graph is an encapsulation
  * of a base graph of type [[B]] with
  * a set of predifined behaviors.
  *
  * @tparam B The type of the encapsulated
  *           base graph object
  */
trait NeuralGraph[B] {
  protected val g: B
  val num_inputs: Int
  val num_outputs: Int
  val forwardPass: (DenseVector[Double]) => DenseVector[Double]
}

object NeuralGraph {

}
