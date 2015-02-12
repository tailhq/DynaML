package org.kuleuven.esat.optimization

import breeze.linalg.{Tensor, Matrix, DenseVector}

/**
 * Trait for optimization problem solvers.
 */
trait Optimizer[T, K, P <: Tensor[K, Double]] extends Serializable {

  /**
   * Solve the convex optimization problem.
   */
  def optimize(data: T, nPoints: Int): P
}
