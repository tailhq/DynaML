package org.kuleuven.esat.optimization

import breeze.linalg._
import breeze.linalg.max

import scala.math._

/**
 * 
 */
abstract class Updater {
  def compute(
      weightsOld: DenseVector[Double],
      gradient: DenseVector[Double],
      stepSize: Double,
      iter: Int,
      regParam: Double): (DenseVector[Double], Double)
}



class SimpleUpdater extends Updater {
  override def compute(
      weightsOld: DenseVector[Double],
      gradient: DenseVector[Double],
      stepSize: Double,
      iter: Int,
      regParam: Double): (DenseVector[Double], Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val weights: DenseVector[Double] = weightsOld
    axpy(-thisIterStepSize, gradient, weights)

    (weights, 0)
  }
}

/**
 * Updater for L1 regularized problems.
 *          R(w) = ||w||_1
 * Uses a step-size decreasing with the square root of the number of iterations.

 * Instead of subgradient of the regularizer, the proximal operator for the
 * L1 regularization is applied after the gradient step. This is known to
 * result in better sparsity of the intermediate solution.
 *
 * The corresponding proximal operator for the L1 norm is the soft-thresholding
 * function. That is, each weight component is shrunk towards 0 by shrinkageVal.
 *
 * If w >  shrinkageVal, set weight component to w-shrinkageVal.
 * If w < -shrinkageVal, set weight component to w+shrinkageVal.
 * If -shrinkageVal < w < shrinkageVal, set weight component to 0.
 *
 * Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
 */
class L1Updater extends Updater {
  override def compute(
      weightsOld: DenseVector[Double],
      gradient: DenseVector[Double],
      stepSize: Double,
      iter: Int,
      regParam: Double): (DenseVector[Double], Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    // Take gradient step
    val weights: DenseVector[Double] = weightsOld
    axpy(-thisIterStepSize, gradient, weights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    while (i < weights.length) {
      val wi = weights(i)
      weights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }

    (weights, norm(weights, 1.0) * regParam)
  }
}

/**
 * :: DeveloperApi ::
 * Updater for L2 regularized problems.
 *          R(w) = 1/2 ||w||**2
 * Uses a step-size decreasing with the square root of the number of iterations.
 */

class SquaredL2Updater extends Updater {
  override def compute(
      weightsOld: DenseVector[Double],
      gradient: DenseVector[Double],
      stepSize: Double,
      iter: Int,
      regParam: Double): (DenseVector[Double], Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val weights: DenseVector[Double] = weightsOld
    weights :*= (1.0 - thisIterStepSize * regParam)
    axpy(-thisIterStepSize, gradient, weights)
    val mag = norm(weights, 2.0)

    (weights, 0.5 * regParam * mag * mag)
  }
}
