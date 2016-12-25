package io.github.mandar2812.dynaml.analysis

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * A [[DataPipe]] which represents a differentiable transformation.
  * */
trait DifferentiableMap[S, D, J] extends DataPipe[S, D] {

  /**
    * Returns the Jacobian of the transformation
    * at the point x.
    * */
  def J(x: S): J
}

object DifferentiableMap {
  def apply[S, D, J](f: (S) => D, j: (S) => J): DifferentiableMap[S, D, J] =
    new DifferentiableMap[S, D, J] {
      /**
        * Returns the Jacobian of the transformation
        * at the point x.
        **/
      override def J(x: S) = j(x)

      override def run(data: S) = f(data)
    }
}