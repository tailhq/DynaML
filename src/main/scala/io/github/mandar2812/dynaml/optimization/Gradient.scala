package io.github.mandar2812.dynaml.optimization

import breeze.linalg._
import org.kuleuven.esat.utils

trait GeneralGradient {
  def compute(
               data: DenseVector[Double],
               label: Double,
               weights: DenseVector[Double])
  : (DenseVector[Double], Double)
}


/**
 * Class used to compute the gradient for a loss function, given a single data point.
 */

abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: DenseVector[Double], loss: Double)
   */
  def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double])
  : (DenseVector[Double], Double)

  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided DenseVector[Double] to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this DenseVector[Double]
   *
   * @return loss
   */
  def compute(
      data: DenseVector[Double],
      label: Double, weights: DenseVector[Double],
      cumGradient: DenseVector[Double]): Double
}

/**
 * Compute gradient and loss for a logistic loss function,
 * as used in binary classification.
 *
 */

class LogisticGradient extends Gradient {
  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double])
  : (DenseVector[Double], Double) = {
    val margin = -1.0 * (weights.t * data)
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
    val gradient = data.copy
    gradient :*= gradientMultiplier
    val loss =
      if (label > 0) {
        // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
        utils.log1pExp(margin)
      } else {
        utils.log1pExp(margin) - margin
      }

    (gradient, loss)
  }

  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double],
      cumGradient: DenseVector[Double]): Double = {
    val margin = -1.0 * (weights.t * data)
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
    axpy(gradientMultiplier, data, cumGradient)
    if (label > 0) {
      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      utils.log1pExp(margin)
    } else {
      utils.log1pExp(margin) - margin
    }
  }
}

/**
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/2 ||weights . phi(x) - y||**2
 */

class LeastSquaresGradient extends Gradient {
  override def compute(
      data: DenseVector[Double],
      label: Double, weights: DenseVector[Double])
  : (DenseVector[Double], Double) = {
    val diff = label - (weights.t * data)//(weights.t * data) - label
    val loss = diff * diff / 2.0
    val gradient = data.copy
    gradient :*= -1*diff
    (gradient, loss)
  }

  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double],
      cumGradient: DenseVector[Double]): Double = {
    val diff = label - (weights.t * data)
    axpy(-1*diff, data, cumGradient)
    diff * diff / 2.0
  }
}

/**
 * Compute gradient and loss for a Hinge loss function, as used in SVM binary classification.
 * NOTE: This assumes that the labels are {0,1}
 */

class HingeGradient extends Gradient {
  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double])
  : (DenseVector[Double], Double) = {
    val dotProduct = weights.t * data
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      val gradient = data.copy
      gradient :*= -labelScaled
      (gradient, 1.0 - labelScaled * dotProduct)
    } else {
      (DenseVector.tabulate(weights.size)((_) => {0.0}), 0.0)
    }
  }

  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double],
      cumGradient: DenseVector[Double]): Double = {
    val dotProduct = weights.t * data
    // Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      axpy(-labelScaled, data, cumGradient)
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }
}

/**
 * Compute gradient and loss for a Least-squared loss function, as used in LS SVM.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/2 (1 - y * weights dot x)**2
 * See also the documentation for the precise formulation.
 */

class LeastSquaresSVMGradient extends Gradient {
  override def compute(
      data: DenseVector[Double],
      label: Double, weights: DenseVector[Double])
  : (DenseVector[Double], Double) = {
    val diff = 1.0 - label*(weights dot data)
    val loss = diff * diff / 2.0
    val gradient = data.copy
    gradient :*= -1*label*diff
    (gradient, loss)
  }

  override def compute(
      data: DenseVector[Double],
      label: Double,
      weights: DenseVector[Double],
      cumGradient: DenseVector[Double]): Double = {
    val diff = 1 - label*(weights.t * data)
    axpy(-1*label*diff, data, cumGradient)
    diff * diff / 2.0
  }
}
