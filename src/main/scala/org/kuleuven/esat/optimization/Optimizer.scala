package org.kuleuven.esat.optimization

import breeze.linalg.{Tensor}
import com.tinkerpop.blueprints.Edge

/**
 * Trait for optimization problem solvers.
 *
 * @tparam K The type indexing the Parameter vector, should be Int in
 *           most cases.
 * @tparam P The type of the parameters of the model to be optimized.
 * @tparam Q The type of the predictor variable
 * @tparam R The type of the target variable
 */
trait Optimizer[K, P <: Tensor[K, Double], Q, R] extends Serializable {

  /**
   * Solve the convex optimization problem.
   */
  def optimize(nPoints: Int, initialP: P,
               ParamOutEdges: java.lang.Iterable[Edge],
               xy: (Edge) => (Q, R)): P
}
