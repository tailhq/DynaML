package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector

/**
  * Created by mandar on 6/2/16.
  */
trait NeuralGraph[B] {
  protected val g: B
  val num_inputs: Int
  val num_outputs: Int
  val forwardPass: (DenseVector[Double]) => DenseVector[Double]
}
