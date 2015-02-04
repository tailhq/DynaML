package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector

/**
 * Trait for optimization problem solvers.
 */
trait Optimizer[T] extends Serializable {

  /**
   * Solve the convex optimization problem.
   */
  def optimize(data: T, nPoints: Int): DenseVector[Double]
}
