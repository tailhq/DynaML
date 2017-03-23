/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.optimization

import breeze.linalg._
import breeze.linalg.max

import scala.math._

trait BasicUpdater[P] extends Serializable {
  def compute(
      weightsOld: P,
      gradient: P,
      stepSize: Double,
      iter: Int,
      regParam: Double): (P, Double)
}

class FFLayerUpdater extends BasicUpdater[Seq[(DenseMatrix[Double], DenseVector[Double])]] {
  override def compute(
    weightsOld: Seq[(DenseMatrix[Double], DenseVector[Double])],
    gradient: Seq[(DenseMatrix[Double], DenseVector[Double])],
    stepSize: Double, iter: Int, regParam: Double) = {
    (
      weightsOld.zip(gradient).map(couple =>
        (
          couple._1._1 - couple._2._1*stepSize - couple._1._1*regParam,
          couple._1._2 - couple._2._2*stepSize - couple._1._2*regParam)),
      0.0)
  }
}

/**
 * 
 */
abstract class Updater
  extends BasicUpdater[DenseVector[Double]]{
  def compute(
      weightsOld: DenseVector[Double],
      gradient: DenseVector[Double],
      stepSize: Double,
      iter: Int,
      regParam: Double): (DenseVector[Double], Double)
}

abstract class HessianUpdater extends Updater {
  def hessianUpdate(oldHessian: DenseMatrix[Double],
                    deltaParams: DenseVector[Double],
                    deltaGradient: DenseVector[Double]): DenseMatrix[Double] = oldHessian
}

class SimpleBFGSUpdater extends HessianUpdater {

  override def compute(weightsOld: DenseVector[Double],
                       gradient: DenseVector[Double],
                       stepSize: Double,
                       iter: Int,
                       regParam: Double): (DenseVector[Double], Double) = {
    (weightsOld + (gradient*stepSize), 0.0)
  }
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
    weights(weights.length-1) = weightsOld(weightsOld.length-1)
    axpy(-thisIterStepSize, gradient, weights)
    val mag = norm(weights, 2.0)

    (weights, 0.5 * regParam * mag * mag)
  }
}
