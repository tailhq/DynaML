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
trait Optimizer[K, P, Q, R, S] extends Serializable {

  /**
   * Solve the convex optimization problem.
   */
  def optimize(nPoints: Long, ParamOutEdges: S, initialP: P): P
}

abstract class RegularizedOptimizer[K, P, Q, R, S]
  extends Optimizer[K, P, Q, R, S] with Serializable {

  protected var regParam: Double = 1.0

  protected var numIterations: Int = 5

  protected var miniBatchFraction: Double = 1.0

  protected var stepSize: Double = 1.0

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

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