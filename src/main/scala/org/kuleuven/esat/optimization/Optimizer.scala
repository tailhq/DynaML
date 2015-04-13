package org.kuleuven.esat.optimization

import breeze.linalg.{Tensor}
import com.tinkerpop.blueprints.Edge
import com.tinkerpop.frames.EdgeFrame

/**
 * Trait for optimization problem solvers.
 *
 * @tparam K The type indexing the Parameter vector, should be Int in
 *           most cases.
 * @tparam P The type of the parameters of the model to be optimized.
 * @tparam Q The type of the predictor variable
 * @tparam R The type of the target variable
 * @tparam S The type of the edge containing the
 *           features and label.
 */
trait Optimizer[K, P <: Tensor[K, Double], Q, R, S] extends Serializable {

  protected var numIterations: Int = 100

  protected var miniBatchFraction: Double = 1.0

  protected var stepSize: Double = 1.0

  /**
   * Solve the convex optimization problem.
   */
  def optimize(nPoints: Int, initialP: P,
               ParamOutEdges: java.lang.Iterable[S]): P

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }
}
